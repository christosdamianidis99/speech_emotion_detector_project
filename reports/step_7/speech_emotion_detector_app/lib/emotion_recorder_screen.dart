import 'dart:convert';
import 'dart:io';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:csv/csv.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:intl/intl.dart';
import 'package:open_file/open_file.dart';
import 'package:path_provider/path_provider.dart';
import 'package:record/record.dart';
import 'package:shared_preferences/shared_preferences.dart';

class EmotionRecorderScreen extends StatefulWidget {
  const EmotionRecorderScreen({Key? key}) : super(key: key);

  @override
  State<EmotionRecorderScreen> createState() => _EmotionRecorderScreenState();
}

class _EmotionRecorderScreenState extends State<EmotionRecorderScreen> {
  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _player = AudioPlayer();

  // State variables
  bool _isRecording = false;
  bool _isLoading = false;
  bool _isPlayingBack = false;

  String? _lastAudioPath;
  List<double>? _waveformData;
  Uint8List? _melImageBytes;

  // Model outputs
  String? _predictedEmotion;
  double? _predictedProb;
  List<MapEntry<String, double>>? _sortedProbs;
  Duration? _inferenceTime;

  // Scientific metrics
  double? _entropyBits;
  double? _normalizedEntropy;
  double? _marginTop1Top2;

  String? _statusText;
  String? _errorMessage;

  // Database (List of records)
  List<EmotionRecord> _dbRecords = [];

  static const Duration _recordDuration = Duration(seconds: 3);
  static const String _baseUrl = 'http://0.0.0.0:8000';

  @override
  void initState() {
    super.initState();
    _loadDatabase();
  }

  @override
  void dispose() {
    _recorder.dispose();
    _player.dispose();
    super.dispose();
  }

  // --- Database Logic ---
  Future<void> _loadDatabase() async {
    final prefs = await SharedPreferences.getInstance();
    final String? jsonString = prefs.getString('emotion_db');
    if (jsonString != null) {
      final List<dynamic> decoded = jsonDecode(jsonString);
      setState(() {
        _dbRecords = decoded.map((e) => EmotionRecord.fromJson(e)).toList();
      });
    }
  }

  Future<void> _saveDatabase() async {
    final prefs = await SharedPreferences.getInstance();
    final String jsonString = jsonEncode(_dbRecords.map((e) => e.toJson()).toList());
    await prefs.setString('emotion_db', jsonString);
  }

  void _addRecordToDb({
    required String predictedEmotion,
    required double confidence,
    required String userActualEmotion,
    required DateTime timestamp,
  }) {
    final newRecord = EmotionRecord(
      timestamp: timestamp,
      predictedEmotion: predictedEmotion,
      confidence: confidence,
      userActualEmotion: userActualEmotion,
    );
    setState(() {
      // Add to beginning of list
      _dbRecords.insert(0, newRecord);
    });
    _saveDatabase();
  }
  // ----------------------

  // --- Export Logic ---
  Future<void> _showExportDialog() async {
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('Export History'),
          content: const Text('Export the recording history as a CSV or JSON file.'),
          actions: [
            TextButton(
              onPressed: () {
                Navigator.pop(context);
                _exportToCsv();
              },
              child: const Text('CSV'),
            ),
            TextButton(
              onPressed: () {
                Navigator.pop(context);
                _exportToJson();
              },
              child: const Text('JSON'),
            ),
          ],
        );
      },
    );
  }

  Future<void> _exportToCsv() async {
    if (_dbRecords.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No data to export.')),
      );
      return;
    }

    final List<List<dynamic>> rows = [];
    rows.add(['Timestamp', 'Predicted Emotion', 'Confidence', 'Actual Emotion']);
    for (final record in _dbRecords) {
      rows.add([
        record.timestamp.toIso8601String(),
        record.predictedEmotion,
        record.confidence,
        record.userActualEmotion,
      ]);
    }

    final String csv = const ListToCsvConverter().convert(rows);
    final String dir = (await getTemporaryDirectory()).path;
    final String path = '$dir/emotion_history.csv';
    final File file = File(path);
    await file.writeAsString(csv);

    OpenFile.open(path);
  }

  Future<void> _exportToJson() async {
    if (_dbRecords.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No data to export.')),
      );
      return;
    }

    // Convert records to a JSON-friendly structure
    final List<Map<String, dynamic>> jsonList = _dbRecords.map((e) {
      return {
        'timestamp': DateFormat('MM/dd/yy HH:mm').format(e.timestamp),
        'predictedEmotion': e.predictedEmotion,
        'confidence': e.confidence, // keep as number, not %
        'actualEmotion': e.userActualEmotion,
      };
    }).toList();

    final String jsonString = jsonEncode(jsonList);

    // Save it to a temporary directory
    final String dir = (await getTemporaryDirectory()).path;
    final String path = '$dir/emotion_history.json';
    final File file = File(path);

    await file.writeAsString(jsonString);

    OpenFile.open(path);
  }
  // ----------------------

  Future<String> _getRecordingPath() async {
    final dir = await getTemporaryDirectory();
    final now = DateTime.now().millisecondsSinceEpoch;
    return '${dir.path}/ser_recording_$now.wav';
  }

  Future<void> _startRecordAndAnalyze() async {
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
      _melImageBytes = null;
      _waveformData = null;
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
    await _loadWaveformData(recordedPath);
    await _playAndThenAnalyze(recordedPath);
  }

  Future<void> _loadWaveformData(String filePath) async {
    try {
      final file = File(filePath);
      final bytes = await file.readAsBytes();
      if (bytes.length <= 44) {
        setState(() { _waveformData = null; });
        return;
      }
      const headerSize = 44;
      final dataBytes = bytes.sublist(headerSize);
      final sampleCount = dataBytes.length ~/ 2;
      final samples = List<double>.filled(sampleCount, 0.0);

      for (int i = 0; i < sampleCount; i++) {
        final lo = dataBytes[2 * i];
        final hi = dataBytes[2 * i + 1];
        int value = (hi << 8) | lo;
        if (value & 0x8000 != 0) { value = value - 0x10000; }
        samples[i] = value / 32768.0;
      }

      const targetPoints = 400;
      if (samples.length <= targetPoints) {
        setState(() { _waveformData = samples; });
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
      setState(() { _waveformData = down; });
    } catch (e) {
      setState(() { _waveformData = null; });
    }
  }

  Future<void> _playAndThenAnalyze(String filePath) async {
    setState(() {
      _isPlayingBack = true;
      _statusText = 'Playing back your recording...';
    });

    try {
      await _player.stop();
      await _player.play(DeviceFileSource(filePath));
      await _player.onPlayerComplete.first;
    } catch (e) {
      setState(() {
        _errorMessage = 'Playback issue (continuing): $e';
      });
    } finally {
      setState(() {
        _isPlayingBack = false;
      });
    }

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
        final allProbsRaw = decoded['all_probs'] as Map<String, dynamic>;
        Uint8List? melImageBytes;
        if (decoded.containsKey('mel_image') && decoded['mel_image'] != null) {
          melImageBytes = base64Decode(decoded['mel_image'] as String);
        }

        final entries = allProbsRaw.entries
            .map((e) => MapEntry<String, double>(e.key, (e.value as num).toDouble()))
            .toList()
          ..sort((a, b) => b.value.compareTo(a.value));

        // Metrics
        final entropy = _computeEntropyBits(entries);
        final maxEntropy = _maxEntropyBits(entries.length);
        final normalizedEntropy = maxEntropy > 0 ? (entropy / maxEntropy).clamp(0.0, 1.0) : 0.0;
        final margin = entries.length >= 2 ? (entries[0].value - entries[1].value) : 0.0;

        setState(() {
          _predictedEmotion = emotion;
          _predictedProb = prob;
          _sortedProbs = entries;
          _inferenceTime = duration;
          _entropyBits = entropy;
          _normalizedEntropy = normalizedEntropy;
          _marginTop1Top2 = margin;
          _melImageBytes = melImageBytes;
          _statusText = null;
        });

        // Prompt user for their actual emotion
        _showActualEmotionDialog(emotion, prob);

      } else {
        setState(() {
          _errorMessage = 'Server error: ${response.statusCode} ${response.reasonPhrase}';
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

  double _computeEntropyBits(List<MapEntry<String, double>> probs) {
    const double eps = 1e-12;
    double h = 0.0;
    for (final e in probs) {
      final p = e.value.clamp(eps, 1.0);
      h -= p * (math.log(p) / math.log(2));
    }
    return h;
  }

  double _maxEntropyBits(int numClasses) {
    if (numClasses <= 1) return 0.0;
    return math.log(numClasses) / math.log(2);
  }

  void _showActualEmotionDialog(String predicted, double confidence) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.white,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
      ),
      builder: (context) {
        return Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text(
                'What was your actual emotion?',
                style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 16),
              Wrap(
                spacing: 12,
                runSpacing: 12,
                alignment: WrapAlignment.center,
                children: ['angry', 'happy', 'neutral', 'sad'].map((e) {
                  return ActionChip(
                    label: Text(e.toUpperCase()),
                    backgroundColor: Colors.grey[100],
                    onPressed: () {
                      Navigator.pop(context);
                      _addRecordToDb(
                        predictedEmotion: predicted,
                        confidence: confidence,
                        userActualEmotion: e,
                        timestamp: DateTime.now(),
                      );
                    },
                  );
                }).toList(),
              ),
              const SizedBox(height: 12),
            ],
          ),
        );
      },
    );
  }

  // --- UI Build Methods ---

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[50],
      appBar: AppBar(
        title: const Text('SER Demo', style: TextStyle(color: Colors.black87, fontWeight: FontWeight.bold)),
        centerTitle: true,
        backgroundColor: Colors.white,
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.ios_share, color: Colors.black54),
            onPressed: _showExportDialog,
            tooltip: 'Export History',
          ),
        ],
      ),
      body: Stack(
        children: [
          Column(
            children: [
              // TOP SECTION: Mic, Waveform
              Container(
                color: Colors.white,
                padding: const EdgeInsets.only(bottom: 24),
                child: Column(
                  children: [
                    const SizedBox(height: 20),
                    SizedBox(
                      height: 80,
                      child: _isRecording
                          ? const RecordingWaveform()
                          : Center(
                        child: Text(
                          'Tap below to record (~3s)',
                          style: TextStyle(color: Colors.grey[600]),
                        ),
                      ),
                    ),
                    const SizedBox(height: 20),
                    _buildRecordButton(),
                    if (_statusText != null) ...[
                      const SizedBox(height: 16),
                      Text(
                        _statusText!,
                        style: const TextStyle(color: Colors.blueGrey),
                      ),
                    ],
                  ],
                ),
              ),

              const Divider(height: 1),

              // BOTTOM SECTION: Scrollable List of Results
              Expanded(
                child: _dbRecords.isEmpty
                    ? Center(
                  child: Text(
                    'No recordings yet.',
                    style: TextStyle(color: Colors.grey[400]),
                  ),
                )
                    : ListView.builder(
                  padding: const EdgeInsets.all(16),
                  itemCount: _dbRecords.length + 1, // +1 for current result header if needed
                  itemBuilder: (context, index) {
                    // Optional: Show the very last result prominently at the top if needed.
                    // Here we just list the DB records.
                    if (index == 0) {
                      // If we have a fresh prediction that might not be in DB yet (e.g. before user selection),
                      // we display it. But our logic adds to DB immediately after selection.
                      // So let's just display the detailed result of the LATEST record if available.
                      if (_predictedEmotion != null) {
                        return Column(
                          children: [
                            _buildCurrentResultCard(),
                            const SizedBox(height: 24),
                            const Align(
                              alignment: Alignment.centerLeft,
                              child: Text(
                                'History',
                                style: TextStyle(
                                    fontSize: 18, fontWeight: FontWeight.bold, color: Colors.black87),
                              ),
                            ),
                            const SizedBox(height: 12),
                          ],
                        );
                      } else {
                         return const SizedBox.shrink();
                      }
                    }
                    final record = _dbRecords[index - 1];
                    return _buildHistoryItem(record);
                  },
                ),
              ),
            ],
          ),

          if (_isLoading)
            Container(
              color: Colors.black45,
              child: const Center(
                child: Card(
                  child: Padding(
                    padding: EdgeInsets.all(24.0),
                    child: CircularProgressIndicator(),
                  ),
                ),
              ),
            ),
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
        width: 72,
        height: 72,
        decoration: BoxDecoration(
          color: _isRecording ? Colors.redAccent : (_isPlayingBack ? Colors.orange : Colors.indigo),
          shape: BoxShape.circle,
          boxShadow: [
            BoxShadow(
              color: (_isRecording ? Colors.redAccent : Colors.indigo).withOpacity(0.3),
              blurRadius: 12,
              spreadRadius: 4,
            )
          ],
        ),
        child: Icon(
          _isRecording ? Icons.stop : (_isPlayingBack ? Icons.volume_up : Icons.mic),
          color: Colors.white,
          size: 32,
        ),
      ),
    );
  }

  Widget _buildCurrentResultCard() {
    if (_predictedEmotion == null) return const SizedBox.shrink();
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
      child: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                const Text('Latest Result', style: TextStyle(fontWeight: FontWeight.bold, color: Colors.grey)),
                const Spacer(),
                if (_inferenceTime != null)
                  Text('${_inferenceTime!.inMilliseconds} ms', style: const TextStyle(fontSize: 10, color: Colors.grey)),
              ],
            ),
            const SizedBox(height: 12),
            Row(
              children: [
                Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      _predictedEmotion!.toUpperCase(),
                      style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.indigo),
                    ),
                    Text(
                      'Confidence: ${(_predictedProb! * 100).toStringAsFixed(1)}%',
                      style: const TextStyle(fontSize: 14, color: Colors.black54),
                    ),
                  ],
                ),
                const Spacer(),
                if (_melImageBytes != null)
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: Image.memory(_melImageBytes!, width: 60, height: 60, fit: BoxFit.cover),
                  ),
              ],
            ),
            if (_sortedProbs != null) ...[
              const SizedBox(height: 16),
              EmotionProbabilitiesView(probs: _sortedProbs!),
            ],
            if (_entropyBits != null) ...[
              const SizedBox(height: 12),
              Text(
                'Entropy: ${_entropyBits!.toStringAsFixed(2)} bits',
                style: const TextStyle(fontSize: 12, color: Colors.grey),
              ),
            ]
          ],
        ),
      ),
    );
  }

  Widget _buildHistoryItem(EmotionRecord record) {
    final isMatch = record.predictedEmotion.toLowerCase() == record.userActualEmotion.toLowerCase();
    final dateStr = DateFormat('MM/dd HH:mm').format(record.timestamp);

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      elevation: 1,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: ListTile(
        leading: CircleAvatar(
          backgroundColor: isMatch ? Colors.green[100] : Colors.orange[100],
          child: Icon(
            isMatch ? Icons.check : Icons.close,
            color: isMatch ? Colors.green : Colors.orange,
            size: 20,
          ),
        ),
        title: Text(
          'Pred: ${record.predictedEmotion.toUpperCase()}',
          style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14),
        ),
        subtitle: Text(
          'Actual: ${record.userActualEmotion.toUpperCase()} • ${dateStr}',
          style: const TextStyle(fontSize: 12),
        ),
        trailing: Text(
          '${(record.confidence * 100).toStringAsFixed(0)}%',
          style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.grey),
        ),
      ),
    );
  }
}

// --- Models & Helpers ---

class EmotionRecord {
  final DateTime timestamp;
  final String predictedEmotion;
  final double confidence;
  final String userActualEmotion;

  EmotionRecord({
    required this.timestamp,
    required this.predictedEmotion,
    required this.confidence,
    required this.userActualEmotion,
  });

  Map<String, dynamic> toJson() => {
    'timestamp': timestamp.toIso8601String(),
    'predictedEmotion': predictedEmotion,
    'confidence': confidence,
    'userActualEmotion': userActualEmotion,
  };

  factory EmotionRecord.fromJson(Map<String, dynamic> json) {
    return EmotionRecord(
      timestamp: DateTime.parse(json['timestamp']),
      predictedEmotion: json['predictedEmotion'],
      confidence: (json['confidence'] as num).toDouble(),
      userActualEmotion: json['userActualEmotion'],
    );
  }
}

class RecordingWaveform extends StatefulWidget {
  const RecordingWaveform({Key? key}) : super(key: key);

  @override
  State<RecordingWaveform> createState() => _RecordingWaveformState();
}

class _RecordingWaveformState extends State<RecordingWaveform> with SingleTickerProviderStateMixin {
  late AnimationController _controller;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(vsync: this, duration: const Duration(milliseconds: 1200))..repeat();
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
            final height = 20 + (val.abs() * 30);
            return Container(
              width: 6,
              height: height,
              margin: const EdgeInsets.symmetric(horizontal: 3),
              decoration: BoxDecoration(
                color: Colors.indigoAccent.withOpacity(0.6),
                borderRadius: BorderRadius.circular(10),
              ),
            );
          }),
        );
      },
    );
  }
}

class EmotionProbabilitiesView extends StatelessWidget {
  final List<MapEntry<String, double>> probs;
  const EmotionProbabilitiesView({Key? key, required this.probs}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: probs.take(4).map((e) {
        final pct = (e.value * 100).toStringAsFixed(1);
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 2.0),
          child: Row(
            children: [
              SizedBox(width: 60, child: Text(e.key, style: const TextStyle(fontSize: 12))),
              Expanded(
                child: Stack(
                  children: [
                    Container(height: 6, decoration: BoxDecoration(color: Colors.grey[200], borderRadius: BorderRadius.circular(3))),
                    FractionallySizedBox(
                      widthFactor: e.value.clamp(0.0, 1.0),
                      child: Container(height: 6, decoration: BoxDecoration(color: Colors.indigoAccent, borderRadius: BorderRadius.circular(3))),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 8),
              SizedBox(width: 40, child: Text('$pct%', style: const TextStyle(fontSize: 10, color: Colors.grey))),
            ],
          ),
        );
      }).toList(),
    );
  }
}
