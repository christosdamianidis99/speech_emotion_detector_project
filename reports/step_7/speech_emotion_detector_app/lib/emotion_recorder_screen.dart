import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

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
  // For visualising the actual recorded signal
  String? _lastAudioPath;
  List<double>? _waveformData; // normalized samples [-1, 1]
  Uint8List? _melImageBytes;  // decoded mel spectrogram image from server

  bool _isRecording = false;
  bool _isLoading = false;
  bool _isPlayingBack = false;

  // Model outputs for the most recent trial
  String? _predictedEmotion;
  double? _predictedProb;
  List<MapEntry<String, double>>? _sortedProbs;
  Duration? _inferenceTime;

  // Derived “scientific” metrics for the most recent trial
  double? _entropyBits;
  double? _normalizedEntropy;
  double? _marginTop1Top2; // difference between top-1 and top-2 probs

  String? _statusText;
  String? _errorMessage;

  // History of predictions (session-level experiment log)
  final List<_PredictionRecord> _history = [];

  // Fixed recording duration (must match MAX_DURATION used in Python)
  static const Duration _recordDuration = Duration(seconds: 3);

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
    // We use WAV so that librosa can read it easily
    return '${dir.path}/ser_recording_$now.wav';
  }

  Future<void> _startRecordAndAnalyze() async {
    // Block interaction while recording, playing back or loading
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

    // Configure WAV at 16 kHz to match the model
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

    // Wait fixed duration, then stop and send
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

    // Store path and compute waveform for visualization
    _lastAudioPath = recordedPath;
    await _loadWaveformData(recordedPath);

    // New: playback first, then analyze
    await _playAndThenAnalyze(recordedPath);
  }
  Future<void> _loadWaveformData(String filePath) async {
    try {
      final file = File(filePath);
      final bytes = await file.readAsBytes();

      // Very simple WAV parser assuming standard 44-byte header, 16-bit PCM, mono.
      if (bytes.length <= 44) {
        setState(() {
          _waveformData = null;
        });
        return;
      }

      const headerSize = 44;
      final dataBytes = bytes.sublist(headerSize);

      // Interpret as little-endian 16-bit signed PCM
      final sampleCount = dataBytes.length ~/ 2;
      final samples = List<double>.filled(sampleCount, 0.0);

      for (int i = 0; i < sampleCount; i++) {
        final lo = dataBytes[2 * i];
        final hi = dataBytes[2 * i + 1];
        int value = (hi << 8) | lo;
        if (value & 0x8000 != 0) {
          value = value - 0x10000;
        }
        samples[i] = value / 32768.0; // normalize to [-1, 1]
      }

      // Downsample for drawing (e.g., to 400 points)
      const targetPoints = 400;
      if (samples.length <= targetPoints) {
        setState(() {
          _waveformData = samples;
        });
        return;
      }

      final step = samples.length / targetPoints;
      final down = <double>[];
      for (int i = 0; i < targetPoints; i++) {
        final idx = (i * step).floor();
        if (idx >= 0 && idx < samples.length) {
          down.add(samples[idx]);
        }
      }

      setState(() {
        _waveformData = down;
      });
    } catch (e) {
      setState(() {
        _waveformData = null;
      });
    }
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
      // Wait until playback completes
      await _player.onPlayerComplete.first;
    } catch (e) {
      // Even if playback fails, we still try to analyze
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

      // Compute duration
      final duration = DateTime.now().difference(startedAt);

      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body) as Map<String, dynamic>;

        final emotion = decoded['emotion'] as String;
        final prob = (decoded['probability'] as num).toDouble();
        final allProbsRaw = decoded['all_probs'] as Map<String, dynamic>;
        Uint8List? melImageBytes;
        if (decoded.containsKey('mel_image') && decoded['mel_image'] != null) {
          final melBase64 = decoded['mel_image'] as String;
          melImageBytes = base64Decode(melBase64);
        }

        // Convert to sorted list of (emotion, prob)
        final entries = allProbsRaw.entries
            .map((e) =>
            MapEntry<String, double>(e.key, (e.value as num).toDouble()))
            .toList()
          ..sort((a, b) => b.value.compareTo(a.value));

        // Compute derived metrics
        final entropy = _computeEntropyBits(entries);
        final maxEntropy = _maxEntropyBits(entries.length);
        final normalizedEntropy =
        maxEntropy > 0 ? (entropy / maxEntropy).clamp(0.0, 1.0) : 0.0;
        final margin = entries.length >= 2
            ? (entries[0].value - entries[1].value)
            : 0.0;

        // Build a canonical probs map (for history)
        final probsMap = <String, double>{
          for (final e in entries) e.key: e.value,
        };

        setState(() {
          _predictedEmotion = emotion;
          _predictedProb = prob;
          _sortedProbs = entries;
          _inferenceTime = duration;
          _entropyBits = entropy;
          _normalizedEntropy = normalizedEntropy;
          _marginTop1Top2 = margin;
          _melImageBytes = melImageBytes;
          _statusText = null; // Clear status text when done

          // Append to history (limit to last 20 trials)
          _history.add(
            _PredictionRecord(
              timestamp: DateTime.now(),
              emotion: emotion,
              confidence: prob,
              probs: probsMap,
            ),
          );
          if (_history.length > 20) {
            _history.removeAt(0);
          }
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

  // Shannon entropy in bits for the distribution
  double _computeEntropyBits(List<MapEntry<String, double>> probs) {
    const double eps = 1e-12;
    double h = 0.0;
    for (final e in probs) {
      final p = e.value.clamp(eps, 1.0);
      h -= p * (math.log(p) / math.log(2)); // log base 2
    }
    return h;
  }
  Widget _buildSessionStatsCard() {
    if (_history.isEmpty) return const SizedBox.shrink();

    final n = _history.length;
    double sumConf = 0.0;
    final Map<String, int> counts = {};
    for (final r in _history) {
      sumConf += r.confidence;
      counts[r.emotion] = (counts[r.emotion] ?? 0) + 1;
    }
    final meanConf = sumConf / n;

    // Normalize counts for simple bars
    final maxCount = counts.values.isEmpty ? 1 : counts.values.reduce(math.max);

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(18.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Session Statistics (local)',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              'Number of trials: $n',
              style: const TextStyle(fontSize: 13),
            ),
            Text(
              'Mean confidence (top-1): ${(meanConf * 100).toStringAsFixed(1)}%',
              style: const TextStyle(fontSize: 13),
            ),
            const SizedBox(height: 12),
            const Text(
              'Emotion distribution (this session):',
              style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500),
            ),
            const SizedBox(height: 8),
            Column(
              children: counts.entries.map((e) {
                final label = e.key;
                final c = e.value;
                final frac = c / maxCount;
                return Padding(
                  padding: const EdgeInsets.symmetric(vertical: 2.0),
                  child: Row(
                    children: [
                      SizedBox(
                        width: 80,
                        child: Text(
                          label,
                          style: const TextStyle(fontSize: 12),
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
                              widthFactor: frac.clamp(0.0, 1.0),
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
                      const SizedBox(width: 8),
                      Text(
                        ' $c',
                        style: const TextStyle(fontSize: 12),
                      ),
                    ],
                  ),
                );
              }).toList(),
            ),
          ],
        ),
      ),
    );
  }
  Widget _buildAcousticEvidenceCard() {
    if (_waveformData == null && _melImageBytes == null) {
      return const SizedBox.shrink();
    }

    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(18.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Acoustic Evidence',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 8),
            const Text(
              'The plots below show the raw waveform (amplitude over time) and the '
                  'log-mel spectrogram used as input to the CNN+GRU model.',
              style: TextStyle(fontSize: 13, color: Colors.black87),
            ),
            const SizedBox(height: 16),
            if (_waveformData != null) ...[
              const Text(
                'Waveform',
                style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500),
              ),
              const SizedBox(height: 8),
              SizedBox(
                height: 80,
                child: CustomPaint(
                  painter: _WaveformPainter(samples: _waveformData!),
                  child: Container(),
                ),
              ),
              const SizedBox(height: 16),
            ],
            if (_melImageBytes != null) ...[
              const Text(
                'Log-mel Spectrogram',
                style: TextStyle(fontSize: 13, fontWeight: FontWeight.w500),
              ),
              const SizedBox(height: 8),
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.memory(
                  _melImageBytes!,
                  fit: BoxFit.cover,
                ),
              ),
            ],
          ],
        ),
      ),
    );
  }

  double _maxEntropyBits(int numClasses) {
    if (numClasses <= 1) return 0.0;
    return math.log(numClasses) / math.log(2);
  }
  String _buildTrialExplanation() {
    if (_predictedEmotion == null || _predictedProb == null || _sortedProbs == null) {
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

    final top2 = _sortedProbs!.length >= 2 ? _sortedProbs!.sublist(0, 2) : _sortedProbs!;
    final topLine = top2.map((e) =>
    '${e.key} (${(e.value * 100).toStringAsFixed(1)}%)').join(', ');

    final buffer = StringBuffer();
    buffer.writeln(
        'For this recording, the model predicts $emotion with $confPct% confidence.');
    buffer.writeln(
        'The two most likely classes according to the softmax output are: $topLine.');

    if (marginPct != null) {
      buffer.writeln(
          'The margin between the most likely and the second most likely emotion is $marginPct percentage points.');
    }
    if (entropyStr != null && normStr != null) {
      buffer.writeln(
          'The entropy of the class distribution is $entropyStr bits '
              '(normalized: $normStr% of the maximum possible uncertainty for four classes).');
    }

    buffer.writeln(
        'A lower entropy and a larger margin typically indicate a more confident decision, '
            'whereas similar probabilities across classes would point to higher ambiguity.');

    return buffer.toString();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[100], // Light, neutral background
      appBar: AppBar(
        title: const Text(
          'SER Demo',
          style: TextStyle(color: Colors.black87, fontWeight: FontWeight.bold),
        ),
        centerTitle: true,
        backgroundColor: Colors.transparent,
        elevation: 0,
        iconTheme: const IconThemeData(color: Colors.black87),
      ),
      body: Stack(
        children: [
          // Main Content
          SafeArea(
            child: Column(
              children: [
                const SizedBox(height: 40),
                // Waveform (Animated)
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

                // Record Button
                _buildRecordButton(),

                const SizedBox(height: 40),

                // Status (optional small text)
                if (_statusText != null)
                  Padding(
                    padding: const EdgeInsets.symmetric(horizontal: 24.0),
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

                // Result + scientific views + history
                Expanded(
                  child: SingleChildScrollView(
                    child: _buildResultSection(),
                  ),
                ),
              ],
            ),
          ),

          // Loading Overlay
          if (_isLoading) _buildLoadingOverlay(),
        ],
      ),
    );
  }

  Widget _buildRecordButton() {
    final bool canInteract = !_isRecording && !_isLoading && !_isPlayingBack;

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
              : (_isPlayingBack ? Colors.deepOrange : Colors.indigoAccent),
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
          shape:
          RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 32, horizontal: 48),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: const [
                CircularProgressIndicator(),
                SizedBox(height: 24),
                Text(
                  'Analyzing...',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
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
          style: const TextStyle(color: Colors.red, fontSize: 16),
          textAlign: TextAlign.center,
        ),
      );
    }

    if (_predictedEmotion == null) {
      return const Padding(
        padding: EdgeInsets.symmetric(horizontal: 30),
        child: Text(
          'Press the button, speak for ~3 seconds,\nlisten to the playback and wait for the result.',
          textAlign: TextAlign.center,
          style: TextStyle(color: Colors.grey, fontSize: 16),
        ),
      );
    }

    return Column(
      children: [
        // MAIN RESULT CARD
        Card(
          margin: const EdgeInsets.symmetric(horizontal: 24),
          elevation: 3,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              children: [
                const Text("Detected Emotion",
                    style: TextStyle(fontSize: 14, color: Colors.grey)),
                const SizedBox(height: 8),
                Text(
                  _predictedEmotion!.toUpperCase(),
                  style: const TextStyle(
                      fontSize: 30, fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                Text(
                  "Confidence: ${(_predictedProb! * 100).toStringAsFixed(1)}%",
                  style: TextStyle(
                      fontSize: 16,
                      color: Colors.indigo.shade700,
                      fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 10),
                if (_inferenceTime != null)
                  Text(
                    "Model response time: ${_inferenceTime!.inMilliseconds} ms",
                    style: const TextStyle(color: Colors.grey, fontSize: 12),
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

        // SIMPLE METRICS
        if (_entropyBits != null && _marginTop1Top2 != null)
          Card(
            margin: const EdgeInsets.symmetric(horizontal: 24),
            elevation: 2,
            shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
            child: Padding(
              padding: const EdgeInsets.all(20.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text("Model Confidence Metrics",
                      style:
                      TextStyle(fontSize: 16, fontWeight: FontWeight.w600)),
                  const SizedBox(height: 12),
                  Text("• Entropy: ${_entropyBits!.toStringAsFixed(3)} bits",
                      style: const TextStyle(fontSize: 14)),
                  const SizedBox(height: 4),
                  Text(
                      "• Margin (top-1 minus top-2): ${( _marginTop1Top2! * 100).toStringAsFixed(1)}%",
                      style: const TextStyle(fontSize: 14)),
                ],
              ),
            ),
          ),

        const SizedBox(height: 20),

        // SHORT EXPLANATION
        _buildShortExplanation(),

        const SizedBox(height: 30),
      ],
    );
  }
  Widget _buildShortExplanation() {
    return Card(
      margin: const EdgeInsets.symmetric(horizontal: 24),
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: const Padding(
        padding: EdgeInsets.all(20.0),
        child: Text(
          "How to read the results:\n"
              "• The model predicts one of four emotions based on speech.\n"
              "• Confidence is the softmax probability of the top class.\n"
              "• The distribution shows how likely each emotion was.\n"
              "• Entropy and margin describe how certain the model was.",
          style: TextStyle(fontSize: 13),
        ),
      ),
    );
  }

}

/// A visual animated waveform to show during recording.
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
    // Loop continuously
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
            // Create a wave effect using sine
            final t = _controller.value * 2 * math.pi;
            final offset = index * 0.8;
            final val = math.sin(t + offset); // -1 to 1

            // Map to height 20..60
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

/// Displays all emotion probabilities in a scientific list style.
class EmotionProbabilitiesView extends StatelessWidget {
  final List<MapEntry<String, double>> probs;

  const EmotionProbabilitiesView({Key? key, required this.probs})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
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
                            fontSize: 14, fontWeight: FontWeight.w500),
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

/// Simple record for one prediction, used in history/log and chart.
class _PredictionRecord {
  final DateTime timestamp;
  final String emotion;
  final double confidence;
  final Map<String, double> probs;

  _PredictionRecord({
    required this.timestamp,
    required this.emotion,
    required this.confidence,
    required this.probs,
  });
}

/// Line chart for confidence over trials (session-level).
class ConfidenceHistoryChart extends StatelessWidget {
  final List<_PredictionRecord> history;

  const ConfidenceHistoryChart({Key? key, required this.history})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    if (history.length < 2) {
      return const SizedBox.shrink();
    }
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: SizedBox(
          height: 160,
          child: CustomPaint(
            painter: _ConfidenceHistoryPainter(history: history),
          ),
        ),
      ),
    );
  }
}

class _ConfidenceHistoryPainter extends CustomPainter {
  final List<_PredictionRecord> history;

  _ConfidenceHistoryPainter({required this.history});

  @override
  void paint(Canvas canvas, Size size) {
    final double paddingLeft = 32;
    final double paddingRight = 12;
    final double paddingTop = 16;
    final double paddingBottom = 28;

    final double chartWidth = size.width - paddingLeft - paddingRight;
    final double chartHeight = size.height - paddingTop - paddingBottom;

    final axisPaint = Paint()
      ..color = Colors.grey
      ..strokeWidth = 1;
    final linePaint = Paint()
      ..color = Colors.indigoAccent
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;
    final pointPaint = Paint()
      ..color = Colors.indigoAccent
      ..style = PaintingStyle.fill;

    // Draw axes (0..1 on y, 1..N on x)
    final origin = Offset(paddingLeft, paddingTop + chartHeight);
    final xEnd = Offset(paddingLeft + chartWidth, paddingTop + chartHeight);
    final yEnd = Offset(paddingLeft, paddingTop);

    canvas.drawLine(origin, xEnd, axisPaint); // x-axis
    canvas.drawLine(origin, yEnd, axisPaint); // y-axis

    final int n = history.length;
    if (n == 0) return;

    // Draw horizontal reference lines at 0.5 and 1.0
    void drawRef(double conf, String label) {
      final y = paddingTop + (1.0 - conf) * chartHeight;
      canvas.drawLine(
        Offset(paddingLeft, y),
        Offset(paddingLeft + chartWidth, y),
        axisPaint..color = Colors.grey.withOpacity(0.3),
      );
    }

    drawRef(1.0, '1.0');
    drawRef(0.5, '0.5');

    final path = Path();
    for (int i = 0; i < n; i++) {
      final double conf = history[i].confidence.clamp(0.0, 1.0);
      final double x = paddingLeft +
          (n == 1 ? chartWidth / 2 : (i / (n - 1)) * chartWidth);
      final double y = paddingTop + (1.0 - conf) * chartHeight;

      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }

      canvas.drawCircle(Offset(x, y), 3, pointPaint);
    }

    canvas.drawPath(path, linePaint);

    // Optional axis labels
    final textPainter = TextPainter(
      textAlign: TextAlign.center,
      textDirection: TextDirection.ltr,
    );

    // "Trial"
    textPainter.text = const TextSpan(
      text: 'Trial',
      style: TextStyle(fontSize: 10, color: Colors.grey),
    );
    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset(
        paddingLeft + chartWidth / 2 - textPainter.width / 2,
        paddingTop + chartHeight + 8,
      ),
    );

    // "Confidence"
    textPainter.text = const TextSpan(
      text: 'Confidence',
      style: TextStyle(fontSize: 10, color: Colors.grey),
    );
    textPainter.layout();
    textPainter.paint(
      canvas,
      Offset(
        paddingLeft - textPainter.width - 4,
        paddingTop - 4,
      ),
    );
  }

  @override
  bool shouldRepaint(covariant _ConfidenceHistoryPainter oldDelegate) {
    return oldDelegate.history != history;
  }
}

/// Simple textual history of trials for inspection.
class ExperimentHistoryView extends StatelessWidget {
  final List<_PredictionRecord> history;

  const ExperimentHistoryView({Key? key, required this.history})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    final items = history.reversed.toList(); // newest first
    return Card(
      elevation: 2,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Experiment Log (current session)',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 8),
            ...items.map((record) {
              final timeStr =
                  '${record.timestamp.hour.toString().padLeft(2, '0')}:'
                  '${record.timestamp.minute.toString().padLeft(2, '0')}:'
                  '${record.timestamp.second.toString().padLeft(2, '0')}';
              final confPct =
              (record.confidence * 100).toStringAsFixed(1);
              return Padding(
                padding: const EdgeInsets.symmetric(vertical: 4.0),
                child: Row(
                  children: [
                    SizedBox(
                      width: 70,
                      child: Text(
                        timeStr,
                        style: const TextStyle(
                          fontSize: 11,
                          color: Colors.grey,
                        ),
                      ),
                    ),
                    SizedBox(
                      width: 80,
                      child: Text(
                        record.emotion.toUpperCase(),
                        style: const TextStyle(
                          fontSize: 13,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    Expanded(
                      child: Text(
                        'conf: $confPct %',
                        style: const TextStyle(
                          fontSize: 12,
                          color: Colors.black87,
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
class _WaveformPainter extends CustomPainter {
  final List<double> samples;

  _WaveformPainter({required this.samples});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.indigoAccent
      ..strokeWidth = 1;

    final midY = size.height / 2;
    final length = samples.length;
    if (length == 0) return;

    final dx = size.width / (length - 1);

    for (int i = 0; i < length - 1; i++) {
      final x1 = i * dx;
      final x2 = (i + 1) * dx;
      final y1 = midY - samples[i] * (size.height / 2);
      final y2 = midY - samples[i + 1] * (size.height / 2);
      canvas.drawLine(Offset(x1, y1), Offset(x2, y2), paint);
    }

    // midline
    final axisPaint = Paint()
      ..color = Colors.grey.withOpacity(0.4)
      ..strokeWidth = 0.5;
    canvas.drawLine(Offset(0, midY), Offset(size.width, midY), axisPaint);
  }

  @override
  bool shouldRepaint(covariant _WaveformPainter oldDelegate) {
    return oldDelegate.samples != samples;
  }
}
