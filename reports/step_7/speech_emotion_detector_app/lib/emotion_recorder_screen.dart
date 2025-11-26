import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';

class EmotionRecorderScreen extends StatefulWidget {
  const EmotionRecorderScreen({Key? key}) : super(key: key);

  @override
  State<EmotionRecorderScreen> createState() => _EmotionRecorderScreenState();
}

class _EmotionRecorderScreenState extends State<EmotionRecorderScreen> {
  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _player = AudioPlayer();

  bool _isRecording = false;
  bool _isLoading = false;
  bool _isPlayingBack = false;

  String? _lastAudioPath;

  // Model outputs
  String? _predictedEmotion;
  double? _predictedProb;
  List<MapEntry<String, double>>? _sortedProbs;
  Duration? _inferenceTime;

  // Confidence / uncertainty metrics
  double? _entropyBits;
  double? _normalizedEntropy;
  double? _marginTop1Top2; // p(top1) - p(top2)

  String? _statusText;
  String? _errorMessage;

  // Fixed recording duration (must match Python MAX_DURATION)
  static const Duration _recordDuration = Duration(seconds: 3);

  // Change to your laptop IP when using a real device
  static const String _baseUrl = 'http://0.0.0.0:8000';

  @override
  void dispose() {
    _recorder.dispose();
    _player.dispose();
    super.dispose();
  }

  Future<String> _getRecordingPath() async {
    final dir = await getTemporaryDirectory();
    final now = DateTime.now().millisecondsSinceEpoch;
    return '${dir.path}/ser_recording_$now.wav';
  }

  Future<void> _startRecordAndAnalyze() async {
    // Block while busy
    if (_isRecording || _isLoading || _isPlayingBack) return;

    setState(() {
      _errorMessage = null;
      _predictedEmotion = null;
      _predictedProb = null;
      _sortedProbs = null;
      _inferenceTime = null;
      _entropyBits = null;
      _normalizedEntropy = null;
      _marginTop1Top2 = null;
      _statusText = 'Requesting microphone permission...';
    });

    if (!await _recorder.hasPermission()) {
      setState(() {
        _statusText = null;
        _errorMessage = 'Microphone permission was not granted.';
      });
      return;
    }

    final path = await _getRecordingPath();

    const config = RecordConfig(
      encoder: AudioEncoder.wav,
      sampleRate: 16000,
      numChannels: 1,
    );

    await _recorder.start(config, path: path);

    setState(() {
      _isRecording = true;
      _statusText = 'Recording (~3 seconds)...';
    });

    // Record fixed duration
    await Future.delayed(_recordDuration);

    final recordedPath = await _recorder.stop();

    setState(() {
      _isRecording = false;
    });

    if (recordedPath == null) {
      setState(() {
        _statusText = null;
        _errorMessage = 'Recording failed.';
      });
      return;
    }

    _lastAudioPath = recordedPath;

    // 1) Playback
    await _playAndThenAnalyze(recordedPath);
  }

  Future<void> _playAndThenAnalyze(String filePath) async {
    // Playback phase
    setState(() {
      _isPlayingBack = true;
      _statusText = 'Playing back your recording...';
    });

    try {
      await _player.stop();
      await _player.play(DeviceFileSource(filePath));
      await _player.onPlayerComplete.first;
    } catch (e) {
      // Even if playback fails, continue with analysis
      setState(() {
        _errorMessage =
        'Playback issue (continuing with analysis): $e';
      });
    } finally {
      setState(() {
        _isPlayingBack = false;
      });
    }

    // Analysis phase
    setState(() {
      _statusText = 'Analyzing...';
    });
    await _sendToServer(filePath);
  }

  Future<void> _sendToServer(String filePath) async {
    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    final startedAt = DateTime.now();

    try {
      final uri = Uri.parse('$_baseUrl/predict');
      final request = http.MultipartRequest('POST', uri)
        ..files.add(await http.MultipartFile.fromPath('file', filePath));

      final streamed = await request.send();
      final response = await http.Response.fromStream(streamed);

      final duration = DateTime.now().difference(startedAt);

      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body) as Map<String, dynamic>;

        final emotion = decoded['emotion'] as String;
        final prob = (decoded['probability'] as num).toDouble();
        final allProbsRaw =
        decoded['all_probs'] as Map<String, dynamic>;

        final entries = allProbsRaw.entries
            .map((e) =>
            MapEntry<String, double>(e.key, (e.value as num).toDouble()))
            .toList()
          ..sort((a, b) => b.value.compareTo(a.value));

        // Confidence metrics: entropy + margin
        final entropy = _computeEntropyBits(entries);
        final maxEntropy = _maxEntropyBits(entries.length);
        final normalizedEntropy =
        maxEntropy > 0 ? (entropy / maxEntropy).clamp(0.0, 1.0) : 0.0;
        final margin = entries.length >= 2
            ? (entries[0].value - entries[1].value)
            : 0.0;

        setState(() {
          _predictedEmotion = emotion;
          _predictedProb = prob;
          _sortedProbs = entries;
          _inferenceTime = duration;
          _entropyBits = entropy;
          _normalizedEntropy = normalizedEntropy;
          _marginTop1Top2 = margin;
          _statusText = null;
        });
      } else {
        setState(() {
          _errorMessage =
          'Server error: ${response.statusCode} ${response.reasonPhrase}';
          _statusText = null;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = 'Could not contact server: $e';
        _statusText = null;
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  // Shannon entropy in bits
  double _computeEntropyBits(List<MapEntry<String, double>> probs) {
    const double eps = 1e-12;
    double h = 0.0;
    for (final e in probs) {
      final p = e.value.clamp(eps, 1.0);
      h -= p * (math.log(p) / math.log(2)); // log2
    }
    return h;
  }

  double _maxEntropyBits(int numClasses) {
    if (numClasses <= 1) return 0.0;
    return math.log(numClasses) / math.log(2);
  }

  String _buildTrialExplanation() {
    if (_predictedEmotion == null ||
        _predictedProb == null ||
        _sortedProbs == null) {
      return 'No prediction available yet.';
    }

    final emotion = _predictedEmotion!;
    final confPct = (_predictedProb! * 100).toStringAsFixed(1);
    final marginPct = _marginTop1Top2 != null
        ? (_marginTop1Top2! * 100).toStringAsFixed(1)
        : null;
    final entropyStr =
    _entropyBits != null ? _entropyBits!.toStringAsFixed(3) : null;
    final normStr = _normalizedEntropy != null
        ? (_normalizedEntropy! * 100).toStringAsFixed(1)
        : null;

    final top2 = _sortedProbs!.length >= 2
        ? _sortedProbs!.sublist(0, 2)
        : _sortedProbs!;
    final topLine = top2
        .map((e) =>
    '${e.key} (${(e.value * 100).toStringAsFixed(1)}%)')
        .join(', ');

    final buffer = StringBuffer();
    buffer.writeln(
        'For this recording, the model predicts $emotion with $confPct% confidence.');
    buffer.writeln(
        'The two most likely classes according to the softmax probabilities are: $topLine.');

    if (marginPct != null) {
      buffer.writeln(
          'The margin between the most likely and the second most likely emotion is $marginPct percentage points. '
              'A larger margin means the model clearly prefers one class over the others.');
    }
    if (entropyStr != null && normStr != null) {
      buffer.writeln(
          'The entropy of the distribution is $entropyStr bits '
              '(about $normStr% of the maximum uncertainty for four classes). '
              'Lower entropy means a more “peaked” distribution, i.e. a more confident decision.');
    }

    buffer.writeln(
        'In the SER literature, entropy and margin are often used to quantify prediction confidence and to identify '
            'ambiguous cases where multiple emotions have similar probability.');

    return buffer.toString();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[100],
      appBar: AppBar(
        title: const Text(
          'SER Demo',
          style: TextStyle(
            color: Colors.black87,
            fontWeight: FontWeight.bold,
          ),
        ),
        centerTitle: true,
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: const IconThemeData(color: Colors.black87),
      ),
      body: Stack(
        children: [
          SafeArea(
            child: Column(
              children: [
                const SizedBox(height: 40),
                SizedBox(
                  height: 100,
                  child: _isRecording
                      ? const RecordingWaveform()
                      : const Center(
                    child: Text(
                      'Ready to record',
                      style: TextStyle(color: Colors.grey),
                    ),
                  ),
                ),
                const SizedBox(height: 40),
                _buildRecordButton(),
                const SizedBox(height: 30),

                if (_statusText != null)
                  Padding(
                    padding:
                    const EdgeInsets.symmetric(horizontal: 24.0),
                    child: Text(
                      _statusText!,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        fontSize: 14,
                        color: Colors.grey,
                      ),
                    ),
                  ),

                const SizedBox(height: 16),

                Expanded(
                  child: SingleChildScrollView(
                    child: _buildResultSection(),
                  ),
                ),
              ],
            ),
          ),

          if (_isLoading) _buildLoadingOverlay(),
        ],
      ),
    );
  }

  Widget _buildRecordButton() {
    final bool canInteract =
        !_isRecording && !_isLoading && !_isPlayingBack;

    return GestureDetector(
      onTap: canInteract ? _startRecordAndAnalyze : null,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        curve: Curves.easeInOut,
        width: 80,
        height: 80,
        decoration: BoxDecoration(
          color: _isRecording
              ? Colors.redAccent
              : (_isPlayingBack
              ? Colors.deepOrange
              : Colors.indigoAccent),
          shape: BoxShape.circle,
          boxShadow: [
            BoxShadow(
              color: (_isRecording
                  ? Colors.redAccent
                  : (_isPlayingBack
                  ? Colors.deepOrange
                  : Colors.indigoAccent))
                  .withOpacity(0.4),
              blurRadius: 16,
              spreadRadius: 4,
            )
          ],
        ),
        child: Icon(
          _isRecording
              ? Icons.stop
              : (_isPlayingBack ? Icons.hearing : Icons.mic),
          color: Colors.white,
          size: 36,
        ),
      ),
    );
  }

  Widget _buildLoadingOverlay() {
    return Container(
      color: Colors.black54,
      child: Center(
        child: Card(
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16)),
          child: Padding(
            padding: const EdgeInsets.symmetric(
                vertical: 32, horizontal: 48),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: const [
                CircularProgressIndicator(),
                SizedBox(height: 24),
                Text(
                  'Analyzing...',
                  style: TextStyle(
                      fontSize: 16, fontWeight: FontWeight.w500),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildResultSection() {
    if (_errorMessage != null) {
      return Padding(
        padding: const EdgeInsets.all(24),
        child: Text(
          _errorMessage!,
          style: const TextStyle(
            color: Colors.red,
            fontSize: 16,
          ),
          textAlign: TextAlign.center,
        ),
      );
    }

    if (_predictedEmotion == null) {
      return const Padding(
        padding: EdgeInsets.symmetric(horizontal: 30),
        child: Text(
          'Press the button, speak for ~3 seconds,\n'
              'listen to the playback and wait for the result.',
          textAlign: TextAlign.center,
          style: TextStyle(
            color: Colors.grey,
            fontSize: 16,
          ),
        ),
      );
    }

    return Column(
      children: [
        // MAIN RESULT
        Card(
          margin: const EdgeInsets.symmetric(horizontal: 24),
          elevation: 3,
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16)),
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              children: [
                const Text(
                  "Detected Emotion",
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.grey,
                  ),
                ),
                const SizedBox(height: 8),
                Text(
                  _predictedEmotion!.toUpperCase(),
                  style: const TextStyle(
                    fontSize: 30,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                if (_predictedProb != null)
                  Text(
                    "Confidence: ${(_predictedProb! * 100).toStringAsFixed(1)}%",
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.indigo.shade700,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                const SizedBox(height: 8),
                if (_inferenceTime != null)
                  Text(
                    "Model response time: ${_inferenceTime!.inMilliseconds} ms",
                    style: const TextStyle(
                      color: Colors.grey,
                      fontSize: 12,
                    ),
                  ),
              ],
            ),
          ),
        ),

        const SizedBox(height: 20),

        // PROBABILITY DISTRIBUTION
        if (_sortedProbs != null)
          EmotionProbabilitiesView(probs: _sortedProbs!),

        const SizedBox(height: 20),

        // METRICS CARD (entropy + margin)
        if (_entropyBits != null && _marginTop1Top2 != null)
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 24),
            elevation: 2,
            shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16)),
            child: Padding(
              padding: const EdgeInsets.all(20.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    "Model Confidence Metrics",
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    "• Entropy: ${_entropyBits!.toStringAsFixed(3)} bits"
                        "${_normalizedEntropy != null ? ' (≈ ${( _normalizedEntropy! * 100).toStringAsFixed(1)}% of maximum uncertainty)' : ''}",
                    style: const TextStyle(fontSize: 14),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    "• Margin (top-1 minus top-2): ${( _marginTop1Top2! * 100).toStringAsFixed(1)}%",
                    style: const TextStyle(fontSize: 14),
                  ),
                  const SizedBox(height: 12),
                  const Text(
                    "These two values are standard in SER research to describe how confident "
                        "the model is. Lower entropy and a larger margin usually mean a clearer decision.",
                    style: TextStyle(fontSize: 13, color: Colors.black87),
                  ),
                ],
              ),
            ),
          ),

        const SizedBox(height: 20),

        // SHORT EDUCATIONAL EXPLANATION (text)
        Card(
          margin: const EdgeInsets.symmetric(horizontal: 24),
          elevation: 1,
          shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(16)),
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Text(
              _buildTrialExplanation(),
              style: const TextStyle(fontSize: 13),
            ),
          ),
        ),

        const SizedBox(height: 30),
      ],
    );
  }
}

/// Animated “fake” waveform while recording (visual feedback)
class RecordingWaveform extends StatefulWidget {
  const RecordingWaveform({Key? key}) : super(key: key);

  @override
  State<RecordingWaveform> createState() => _RecordingWaveformState();
}

class _RecordingWaveformState extends State<RecordingWaveform>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1200),
    )..repeat();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Row(
          mainAxisAlignment: MainAxisAlignment.center,
          crossAxisAlignment: CrossAxisAlignment.center,
          children: List.generate(7, (index) {
            final t = _controller.value * 2 * math.pi;
            final offset = index * 0.8;
            final val = math.sin(t + offset);
            final height = 20 + (val.abs() * 40);

            return Container(
              width: 8,
              height: height,
              margin: const EdgeInsets.symmetric(horizontal: 4),
              decoration: BoxDecoration(
                color: Colors.indigoAccent.withOpacity(0.7),
                borderRadius: BorderRadius.circular(10),
              ),
            );
          }),
        );
      },
    );
  }
}

/// Simple probability bars for each emotion.
class EmotionProbabilitiesView extends StatelessWidget {
  final List<MapEntry<String, double>> probs;

  const EmotionProbabilitiesView({Key? key, required this.probs})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 24),
      elevation: 2,
      shape:
      RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Class Probabilities',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 16),
            ...probs.map((e) {
              final pct = (e.value * 100).toStringAsFixed(1);
              return Padding(
                padding: const EdgeInsets.symmetric(vertical: 4.0),
                child: Row(
                  children: [
                    SizedBox(
                      width: 80,
                      child: Text(
                        e.key,
                        style: const TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                    Expanded(
                      child: Stack(
                        children: [
                          Container(
                            height: 8,
                            decoration: BoxDecoration(
                              color: Colors.grey[200],
                              borderRadius: BorderRadius.circular(4),
                            ),
                          ),
                          FractionallySizedBox(
                            widthFactor: e.value.clamp(0.0, 1.0),
                            child: Container(
                              height: 8,
                              decoration: BoxDecoration(
                                color: Colors.indigoAccent,
                                borderRadius: BorderRadius.circular(4),
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                    const SizedBox(width: 12),
                    SizedBox(
                      width: 60,
                      child: Text(
                        '$pct %',
                        textAlign: TextAlign.end,
                        style: TextStyle(
                          fontSize: 12,
                          color: Colors.grey[700],
                        ),
                      ),
                    ),
                  ],
                ),
              );
            }).toList(),
          ],
        ),
      ),
    );
  }
}
