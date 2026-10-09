import 'dart:async';
import 'dart:convert';
import 'dart:isolate';
import 'dart:math' as math;
import 'dart:typed_data';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';
import 'package:image/image.dart' as img;
import 'package:socket_io_client/socket_io_client.dart' as io;

// =====================================================================
// CONFIG
// =====================================================================

/// One flat "camera" cut out of the 360 frame.
/// yawDeg: 0 = straight ahead, 90 = right, 180 = behind.
/// latDeg: 0 = horizon, +90 = straight up, -90 = straight down.
class ViewSpec {
  const ViewSpec(this.yawDeg, this.latDeg);
  final double yawDeg;
  final double latDeg;

  double get yaw => yawDeg * math.pi / 180;
  double get pitch => -latDeg * math.pi / 180; // the shader's pitch is "down"
}

class ApiConfig {
  /// Same server as NEXT_PUBLIC_YOLO_SERVER_URL in the web app.
  /// Override at build time:
  ///   flutter run --dart-define=YOLO_SERVER_URL=http://192.168.1.20:8000
  static const String serverUrl = String.fromEnvironment(
    'YOLO_SERVER_URL',
    defaultValue: 'http://192.168.0.111:8000',
  );

  static const String sendEvent = 'detect-frame';
  static const String resultEvent = 'detection-result';
  static const String errorEvent = 'detection-error';

  // ---- how the 360 frame is cut up for the detector ----

  /// The detector never sees the stretched 360 picture. The sphere is cut into
  /// normal (perspective) views like a regular camera would see, and each one
  /// is sent as its own frame. Four views 90 degrees apart cover the horizon.
  /// Add ViewSpec(0, 90) and ViewSpec(0, -90) for ceiling and floor.
  /// More views = better coverage, but each cycle takes longer.
  static const List<ViewSpec> views = [
    ViewSpec(0, 0),
    ViewSpec(90, 0),
    ViewSpec(180, 0),
    ViewSpec(270, 0),
  ];

  /// Field of view of each view. Larger than the 90 degree spacing so that
  /// neighbouring views overlap and objects on a seam are seen whole.
  static const double viewFovDeg = 110;

  /// Each view is size x size pixels. Keep it at 640 for a YOLO model that
  /// runs at 640 (the server's mask coordinates then match the image exactly).
  static const int viewSize = 640;

  static const int jpegQuality = 50; // the web version uses 0.5
  static const int minIntervalMs = 0; // minimum gap between requests

  /// How many views are out at the server at the same time. 2 keeps the
  /// server busy while the next view is being rendered and encoded. The
  /// server handles frames one by one, so more than 2-3 only adds waiting.
  static const int maxInFlight = 2;
  static int get inFlightLimit =>
      maxInFlight < views.length ? maxInFlight : views.length;

  /// Background isolates that encode views to JPEG.
  static const int encoderIsolates = 2;
  static const Duration timeout = Duration(seconds: 3);

  /// Outlines older than this are removed from the screen.
  static const int resultTtlMs = 2000;
  static const double minScore = 0.0;

  /// Douglas-Peucker tolerance in pixels of a view (web uses 1.5).
  static const double simplifyEpsilonPx = 2.0;

  /// true = no network, returns a fake circle in the middle of every view.
  /// Use it to check that the outlines line up with the preview.
  static const bool useMock = false;
}

// =====================================================================
// Model
// =====================================================================

class Detection {
  const Detection(this.label, this.score, this.points);
  final String label;
  final double score;

  /// Outline. Points are normalized 0..1 inside the image they came from.
  /// After `mapViewDetection` they are 0..1 across the full 360
  /// (equirectangular) frame: x = 0 is the left edge, y = 0 is the top.
  final List<Offset> points;
}

// =====================================================================
// View rendering (GPU) and coordinate mapping
// =====================================================================

/// Renders one perspective view out of the full-resolution 360 frame, using
/// the same shader as the preview. Returns a size x size image.
Future<ui.Image> renderView(
  ui.FragmentShader shader,
  ui.Image frame360,
  ViewSpec v,
) async {
  final size = ApiConfig.viewSize;
  final s = size.toDouble();
  shader
    ..setFloat(0, s)
    ..setFloat(1, s)
    ..setFloat(2, v.yaw)
    ..setFloat(3, v.pitch)
    ..setFloat(4, ApiConfig.viewFovDeg * math.pi / 180)
    ..setImageSampler(0, frame360);

  final rec = ui.PictureRecorder();
  Canvas(rec).drawRect(Rect.fromLTWH(0, 0, s, s), Paint()..shader = shader);
  final pic = rec.endRecording();
  try {
    return await pic.toImage(size, size);
  } finally {
    pic.dispose();
  }
}

/// A pixel in a rendered view -> (u, v) on the 360 frame. This is the shader's
/// math run forwards for one pixel.
Offset viewPixelToUv(
  double px,
  double py,
  double size,
  double yaw,
  double pitch,
  double fov,
) {
  final f = 1.0 / math.tan(fov / 2);
  final nx = (px / size) * 2 - 1;
  final ny = (py / size) * 2 - 1;
  var x = nx, y = -ny, z = f;
  final n = math.sqrt(x * x + y * y + z * z);
  x /= n;
  y /= n;
  z /= n;

  final cp = math.cos(pitch), sp = math.sin(pitch);
  final y1 = y * cp - z * sp;
  final z1 = y * sp + z * cp;
  final cy = math.cos(yaw), sy = math.sin(yaw);
  final x2 = x * cy + z1 * sy;
  final z2 = -x * sy + z1 * cy;

  final lon = math.atan2(x2, z2);
  final lat = math.asin(y1.clamp(-1.0, 1.0).toDouble());
  return Offset(lon / (2 * math.pi) + 0.5, 0.5 - lat / math.pi);
}

/// Moves a detection from view coordinates to 360-frame coordinates.
Detection mapViewDetection(Detection d, ViewSpec v) {
  final size = ApiConfig.viewSize.toDouble();
  final fov = ApiConfig.viewFovDeg * math.pi / 180;
  return Detection(d.label, d.score, [
    for (final p in d.points)
      viewPixelToUv(p.dx * size, p.dy * size, size, v.yaw, v.pitch, fov),
  ]);
}

List<double> _dirOfUv(Offset uv) {
  final lon = (uv.dx - 0.5) * 2 * math.pi;
  final lat = (0.5 - uv.dy) * math.pi;
  final cl = math.cos(lat);
  return [cl * math.sin(lon), math.sin(lat), cl * math.cos(lon)];
}

List<double> _dirOfView(ViewSpec v) {
  final cp = math.cos(v.pitch);
  return [cp * math.sin(v.yaw), -math.sin(v.pitch), cp * math.cos(v.yaw)];
}

List<double> _centroidDir(List<Offset> pts) {
  double x = 0, y = 0, z = 0;
  for (final p in pts) {
    final d = _dirOfUv(p);
    x += d[0];
    y += d[1];
    z += d[2];
  }
  final n = math.sqrt(x * x + y * y + z * z);
  return n == 0 ? [0, 0, 1] : [x / n, y / n, z / n];
}

/// Combines the detections of all views. Neighbouring views overlap, so the
/// same object shows up twice. Each object is kept only from the view whose
/// centre is closest to the object's centre, where distortion is lowest. If
/// that view did not answer, the other views' result is kept instead.
/// `perView[i]` holds view i's detections, already in 360-frame coordinates,
/// or null if that view failed.
List<Detection> mergeViews(List<List<Detection>?> perView) {
  final centers = [for (final v in ApiConfig.views) _dirOfView(v)];
  final out = <Detection>[];
  for (int i = 0; i < perView.length; i++) {
    final list = perView[i];
    if (list == null) continue;
    for (final d in list) {
      final c = _centroidDir(d.points);
      int best = 0;
      double bestDot = -2;
      for (int k = 0; k < centers.length; k++) {
        final dot =
            c[0] * centers[k][0] + c[1] * centers[k][1] + c[2] * centers[k][2];
        if (dot > bestDot) {
          bestDot = dot;
          best = k;
        }
      }
      if (best == i || perView[best] == null) out.add(d);
    }
  }
  return out;
}

// =====================================================================
// JPEG encoding (background isolate) -> "data:image/jpeg;base64,..."
// Same payload the web version sends in `image`.
// =====================================================================

class _JpegJob {
  _JpegJob(this.data, this.w, this.h, this.quality, this.reply);
  final TransferableTypedData data;
  final int w, h, quality;
  final SendPort reply;
}

void _jpegWorkerMain(SendPort ready) {
  final port = ReceivePort();
  ready.send(port.sendPort);
  port.listen((msg) {
    final job = msg as _JpegJob;
    try {
      final bytes = job.data.materialize().asUint8List();
      final im = img.Image.fromBytes(
        width: job.w,
        height: job.h,
        bytes: bytes.buffer,
        bytesOffset: bytes.offsetInBytes,
        numChannels: 4,
      );
      final jpg = img.encodeJpg(im, quality: job.quality);
      job.reply.send(['ok', 'data:image/jpeg;base64,${base64Encode(jpg)}']);
    } catch (e) {
      job.reply.send(['err', '$e']);
    }
  });
}

/// Encoder isolates are started once and reused. Spawning a new isolate for
/// every frame (Isolate.run) costs several milliseconds each time, and the
/// pixels are handed over without copying.
class _JpegPool {
  final List<SendPort> _ports = [];
  final List<Isolate> _isolates = [];
  Future<void>? _starting;
  int _next = 0;

  Future<void> _start() async {
    for (int i = 0; i < ApiConfig.encoderIsolates; i++) {
      final ready = ReceivePort();
      _isolates.add(await Isolate.spawn(_jpegWorkerMain, ready.sendPort));
      _ports.add(await ready.first as SendPort);
      ready.close();
    }
  }

  Future<String> encode(Uint8List rgba, int w, int h) async {
    await (_starting ??= _start());
    final port = _ports[_next++ % _ports.length];
    final reply = ReceivePort();
    port.send(
      _JpegJob(
        TransferableTypedData.fromList([rgba]),
        w,
        h,
        ApiConfig.jpegQuality,
        reply.sendPort,
      ),
    );
    final res = await reply.first as List;
    reply.close();
    if (res[0] != 'ok') throw 'jpeg encode failed: ${res[1]}';
    return res[1] as String;
  }

  void dispose() {
    for (final i in _isolates) {
      i.kill(priority: Isolate.immediate);
    }
    _isolates.clear();
    _ports.clear();
    _starting = null;
  }
}

final _JpegPool _jpegPool = _JpegPool();

Future<String> encodeJpegDataUrl(Uint8List rgba, int w, int h) =>
    _jpegPool.encode(rgba, w, h);

void disposeJpegPool() => _jpegPool.dispose();

// =====================================================================
// Socket.IO client
//   emit   'detect-frame'      { frameId, image: dataUrl }
//   listen 'detection-result'  { frameId, detections: [{label, confidence, bbox, segments}] }
//   listen 'detection-error'
// Several frames (one per view) can be waiting at once. Results are matched
// by frameId, so results meant for other clients are ignored (the server
// broadcasts every result to everybody).
// =====================================================================

class _Pending {
  _Pending(this.completer, this.w, this.h);
  final Completer<List<Detection>> completer;
  final int w, h;
}

class DetectionClient {
  final ValueNotifier<bool> connected = ValueNotifier(false);
  void Function(String message)? onLog;

  /// Makes frame ids unique per app run (the web client also counts from 0).
  final String tag = 'x5${DateTime.now().millisecondsSinceEpoch % 1000000}';

  io.Socket? _socket;
  final Map<String, _Pending> _pending = {}; // insertion order = send order
  bool _closed = false;

  bool get ready => ApiConfig.useMock || connected.value;

  void connect() {
    if (ApiConfig.useMock || _socket != null) return;

    final socket = io.io(
      ApiConfig.serverUrl,
      io.OptionBuilder()
          .setTransports(['websocket'])
          .enableAutoConnect()
          .build(),
    );
    _socket = socket;

    socket.onConnect((_) {
      if (_closed) return;
      connected.value = true;
      onLog?.call('Socket connected: ${ApiConfig.serverUrl}');
    });
    socket.onDisconnect((_) {
      if (_closed) return;
      connected.value = false;
      onLog?.call('Socket disconnected');
      _failAll('socket disconnected');
    });
    socket.onConnectError((e) => onLog?.call('Socket connect error: $e'));

    socket.on(ApiConfig.resultEvent, _onResult);
    socket.on(ApiConfig.errorEvent, (err) => _failAll('server error: $err'));
  }

  void _onResult(dynamic data) {
    dynamic root = data;
    if (root is String) {
      try {
        root = jsonDecode(root);
      } catch (_) {
        return;
      }
    }
    if (root is! Map) return;

    _Pending? p;
    final id = root['frameId']?.toString();
    if (id != null) {
      p = _pending.remove(id);
      if (p == null) return; // not ours
    } else if (root['success'] == false && _pending.isNotEmpty) {
      // The worker reports some errors without a frameId. It answers in the
      // order it received frames, so this belongs to the oldest one.
      p = _pending.remove(_pending.keys.first);
    }
    if (p == null || p.completer.isCompleted) return;

    if (root['success'] == false) {
      p.completer.completeError('server error: ${root['error']}');
      return;
    }
    try {
      p.completer.complete(parseDetections(root, p.w, p.h));
    } catch (e) {
      p.completer.completeError(e);
    }
  }

  void _failAll(String message) {
    final all = _pending.values.toList();
    _pending.clear();
    for (final p in all) {
      if (!p.completer.isCompleted) p.completer.completeError(message);
    }
  }

  /// Sends one frame and waits for its result. The points of the returned
  /// detections are normalized 0..1 inside this frame.
  Future<List<Detection>> detect(
    String frameId,
    String dataUrl,
    int w,
    int h,
  ) async {
    if (ApiConfig.useMock) {
      await Future.delayed(const Duration(milliseconds: 60));
      return [
        Detection('mock', 0.99, [
          for (int i = 0; i < 24; i++)
            Offset(
              0.5 + 0.18 * math.cos(i * math.pi / 12),
              0.5 + 0.18 * math.sin(i * math.pi / 12),
            ),
        ]),
      ];
    }

    final socket = _socket;
    if (socket == null || !connected.value) {
      throw 'server not connected';
    }
    final c = Completer<List<Detection>>();
    _pending[frameId] = _Pending(c, w, h);
    socket.emit(ApiConfig.sendEvent, {'frameId': frameId, 'image': dataUrl});
    return c.future.timeout(
      ApiConfig.timeout,
      onTimeout: () {
        _pending.remove(frameId);
        throw TimeoutException('no response for $frameId');
      },
    );
  }

  void close() {
    _closed = true;
    _failAll('closed');
    _socket?.dispose();
    _socket = null;
    connected.dispose();
  }
}

// =====================================================================
// Response parsing
// =====================================================================

List<Detection> parseDetections(dynamic payload, int sentW, int sentH) {
  dynamic root = payload;
  if (root is String) root = jsonDecode(root);
  final items = root is Map ? root['detections'] : root;
  if (items is! List) return const [];

  final out = <Detection>[];
  for (final it in items) {
    if (it is! Map) continue;
    final conf = it['confidence'] ?? it['score'];
    final score = conf is num ? conf.toDouble() : 0.0;
    if (score < ApiConfig.minScore) continue;
    final label = '${it['label'] ?? 'object'}';

    final pts = _outlineOf(it, sentW, sentH);
    if (pts.length < 3) continue;
    out.add(Detection(label, score, pts));
  }
  return out;
}

/// Builds a normalized outline from `segments` (or from `bbox` if there are
/// no segments). Coordinates up to 1.5 are treated as already normalized,
/// larger values as pixels of the frame that was sent.
List<Offset> _outlineOf(Map it, int w, int h) {
  var pts = _toPoints(
    it['segments'] ??
        it['polygon'] ??
        it['points'] ??
        it['outline'] ??
        it['contour'],
  );
  bool fromBox = false;
  if (pts.length < 3) {
    final b = it['bbox'] ?? it['box'];
    if (b is Map &&
        b['x1'] is num &&
        b['y1'] is num &&
        b['x2'] is num &&
        b['y2'] is num) {
      pts = _boxRing(
        (b['x1'] as num).toDouble(),
        (b['y1'] as num).toDouble(),
        (b['x2'] as num).toDouble(),
        (b['y2'] as num).toDouble(),
      );
      fromBox = true;
    } else {
      return const [];
    }
  }

  double maxV = 0;
  for (final p in pts) {
    maxV = math.max(maxV, math.max(p.dx, p.dy));
  }
  final isPixels = maxV > 1.5;

  if (!fromBox) {
    pts = simplifyPolygon(
      pts,
      isPixels ? ApiConfig.simplifyEpsilonPx : ApiConfig.simplifyEpsilonPx / w,
    );
  }
  return isPixels ? [for (final p in pts) Offset(p.dx / w, p.dy / h)] : pts;
}

// A rectangle with extra points along each edge, so it bends correctly when
// it is projected onto the 360 view.
List<Offset> _boxRing(double x1, double y1, double x2, double y2) {
  const n = 10;
  Offset lerp(Offset a, Offset b, int i) =>
      Offset(a.dx + (b.dx - a.dx) * i / n, a.dy + (b.dy - a.dy) * i / n);
  final tl = Offset(x1, y1), tr = Offset(x2, y1);
  final br = Offset(x2, y2), bl = Offset(x1, y2);
  return [
    for (int i = 0; i < n; i++) lerp(tl, tr, i),
    for (int i = 0; i < n; i++) lerp(tr, br, i),
    for (int i = 0; i < n; i++) lerp(br, bl, i),
    for (int i = 0; i < n; i++) lerp(bl, tl, i),
  ];
}

List<Offset> _toPoints(dynamic raw) {
  if (raw is! List || raw.isEmpty) return const [];
  final first = raw.first;
  if (first is num) {
    return [
      for (int i = 0; i + 1 < raw.length; i += 2)
        Offset((raw[i] as num).toDouble(), (raw[i + 1] as num).toDouble()),
    ];
  }
  if (first is Map) {
    return [
      for (final p in raw)
        if (p is Map && p['x'] is num && p['y'] is num)
          Offset((p['x'] as num).toDouble(), (p['y'] as num).toDouble()),
    ];
  }
  if (first is List) {
    if (first.length == 2 && first[0] is num) {
      return [
        for (final p in raw)
          if (p is List && p.length >= 2 && p[0] is num && p[1] is num)
            Offset((p[0] as num).toDouble(), (p[1] as num).toDouble()),
      ];
    }
    return _toPoints(first); // nested lists: use the first polygon
  }
  return const [];
}

// Douglas-Peucker. Handles a zero-length base line (closed outlines whose
// first and last point are equal), which would otherwise divide by zero.
List<Offset> simplifyPolygon(List<Offset> pts, double eps) {
  if (pts.length < 3) return pts;
  final a = pts.first, b = pts.last;
  double dMax = 0;
  int idx = 0;
  for (int i = 1; i < pts.length - 1; i++) {
    final d = _distToLine(pts[i], a, b);
    if (d > dMax) {
      dMax = d;
      idx = i;
    }
  }
  if (dMax > eps) {
    final l = simplifyPolygon(pts.sublist(0, idx + 1), eps);
    final r = simplifyPolygon(pts.sublist(idx), eps);
    return [...l.sublist(0, l.length - 1), ...r];
  }
  return [a, b];
}

double _distToLine(Offset p, Offset a, Offset b) {
  final dx = b.dx - a.dx, dy = b.dy - a.dy;
  final len2 = dx * dx + dy * dy;
  if (len2 == 0) return (p - a).distance;
  final t = ((p.dx - a.dx) * dx + (p.dy - a.dy) * dy) / len2;
  final px = a.dx + t * dx, py = a.dy + t * dy;
  final ex = p.dx - px, ey = p.dy - py;
  return math.sqrt(ex * ex + ey * ey);
}

// =====================================================================
// Overlay: draws outlines on top of the 360 view.
//
// Points live in equirectangular space, so they are projected with the same
// yaw / pitch / fov as the shader. Outlines stay glued to the objects while
// you pan and zoom.
// =====================================================================

class OutlinePainter extends CustomPainter {
  OutlinePainter(this.dets, this.yaw, this.pitch, this.fov, Listenable repaint)
    : super(repaint: repaint);

  final ValueNotifier<List<Detection>> dets;
  final double Function() yaw;
  final double Function() pitch;
  final double Function() fov;

  final Paint _stroke = Paint()
    ..style = PaintingStyle.stroke
    ..strokeWidth = 2.5
    ..strokeJoin = StrokeJoin.round;
  final Paint _fill = Paint()..style = PaintingStyle.fill;
  final Paint _labelBg = Paint();

  // Label layout is expensive and only changes when the detections change,
  // so it is done once per result list instead of on every repaint.
  List<Detection>? _cachedFor;
  List<TextPainter> _tps = const [];
  List<Color> _colors = const [];

  void _rebuildCache(List<Detection> list) {
    _cachedFor = list;
    for (final t in _tps) {
      t.dispose();
    }
    _tps = [
      for (final d in list)
        TextPainter(
          text: TextSpan(
            text: '${d.label} ${(d.score * 100).round()}%',
            style: const TextStyle(
              color: Colors.white,
              fontSize: 11,
              fontWeight: FontWeight.w600,
            ),
          ),
          textDirection: TextDirection.ltr,
        )..layout(),
    ];
    // Colour follows the label, so it doesn't flicker when the list order
    // changes between updates.
    _colors = [
      for (final d in list)
        _palette[d.label.codeUnits.fold<int>(
              0,
              (a, b) => (a * 31 + b) & 0xffff,
            ) %
            _palette.length],
    ];
  }

  // Same palette as the web version, in order.
  static const _palette = [
    Color(0xFFFF4D4F),
    Color(0xFF4ADE80),
    Color(0xFF38BDF8),
    Color(0xFFF59E0B),
    Color(0xFFC084FC),
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final list = dets.value;
    if (list.isEmpty) return;
    if (!identical(list, _cachedFor)) _rebuildCache(list);

    final y = yaw(), p = pitch();
    final f = 1.0 / math.tan(fov() * 0.5);
    final cy = math.cos(y), sy = math.sin(y);
    final cp = math.cos(p), sp = math.sin(p);
    final aspect = size.width / size.height;

    // Inverse of the shader's mapping: equirect (u,v) -> screen pixel.
    Offset? project(Offset uv) {
      final lon = (uv.dx - 0.5) * 2 * math.pi;
      final lat = (0.5 - uv.dy) * math.pi;
      final cl = math.cos(lat);
      final x = cl * math.sin(lon);
      final yv = math.sin(lat);
      final z = cl * math.cos(lon);
      // undo yaw, then undo pitch
      final x1 = x * cy - z * sy;
      final z1 = x * sy + z * cy;
      final y2 = yv * cp + z1 * sp;
      final z2 = -yv * sp + z1 * cp;
      if (z2 <= 0.02) return null; // behind the camera
      final px = f * x1 / z2;
      final py = -f * y2 / z2;
      return Offset(
        (px / aspect + 1) * 0.5 * size.width,
        (py + 1) * 0.5 * size.height,
      );
    }

    for (int i = 0; i < list.length; i++) {
      final d = list[i];
      final path = Path();
      bool pen = false;
      bool allVisible = true;
      Offset? labelAt;
      for (final uv in d.points) {
        final s = project(uv);
        if (s == null) {
          pen = false;
          allVisible = false;
          continue;
        }
        if (pen) {
          path.lineTo(s.dx, s.dy);
        } else {
          path.moveTo(s.dx, s.dy);
        }
        pen = true;
        if (labelAt == null || s.dy < labelAt.dy) labelAt = s;
      }
      if (labelAt == null) continue; // fully off screen / behind

      final color = _colors[i];
      if (allVisible) {
        path.close();
        _fill.color = color.withAlpha(46);
        canvas.drawPath(path, _fill);
      }
      _stroke.color = color;
      canvas.drawPath(path, _stroke);

      final tp = _tps[i];
      final maxX = math.max(0.0, size.width - tp.width - 8);
      final maxY = math.max(0.0, size.height - tp.height - 2);
      final pos = Offset(
        labelAt.dx.clamp(6.0, maxX + 6).toDouble(),
        (labelAt.dy - tp.height - 4).clamp(0.0, maxY).toDouble(),
      );
      _labelBg.color = const Color(0xB3000000); // rgba(0,0,0,0.7) like the web
      canvas.drawRect(
        Rect.fromLTWH(pos.dx - 6, pos.dy, tp.width + 12, tp.height + 2),
        _labelBg,
      );
      tp.paint(canvas, Offset(pos.dx, pos.dy + 1));
    }
  }

  @override
  bool shouldRepaint(covariant OutlinePainter old) => false; // driven by `repaint`
}
