import 'package:adhd_diagnosis/ui/games/flappy_bird.dart';
import 'package:flame/game.dart';
import 'package:flutter/material.dart';
import './data_controller.dart';
import '../ui/game_screen.dart';
import 'package:flutter/foundation.dart';

class GameLoaderController {
  static Widget loadGame({
    required GameWidget<FlappyBirdGame> game,
    required Function(String) onGazeDetected,
    required Function(String) onError,
  }) {

    return GameScreen(
      game: game,
    );
  }
}
