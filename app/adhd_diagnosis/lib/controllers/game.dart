import 'package:flame/game.dart';
import 'package:flutter/material.dart';
import '../controllers/data_controller.dart';

class ADHDGame extends FlameGame {
  final ValueNotifier<int> score = ValueNotifier<int>(0);
  final DataController dataController;

  ADHDGame({required this.dataController});
}
