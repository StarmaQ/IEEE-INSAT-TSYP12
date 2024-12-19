import 'package:flutter/material.dart';
import 'package:adhd_diagnosis/ui/game_menu/game_menu.dart';
import 'dart:async';

// void main() {
//   runApp(const MyApp());
// }

// class MyApp extends StatelessWidget {
//   const MyApp({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       title: 'Game Menu',
//       theme: ThemeData(
//         primarySwatch: Colors.blue,
//       ),
//       home: const ,
//     );
//   }
// }

class SplashScreen extends StatefulWidget {
  @override
  _SplashScreenState createState() => _SplashScreenState();
}

class _SplashScreenState extends State<SplashScreen> {
  double _opacity = 1.0;

  @override
  void initState() {
    super.initState();

    // Start the fade out after a delay
    Timer(Duration(seconds: 2), () {
      setState(() {
        _opacity = 0.0;
      });

      // After fading out, navigate to the home screen
      Timer(Duration(seconds: 1), () {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(builder: (context) => GameMenuScreen()), // Replace with your home screen widget
        );
      });
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      body: Center(
        child: AnimatedOpacity(
          opacity: _opacity,
          duration: Duration(seconds: 1),
          child: Image.asset('assets/logo.png'),
        ),
      ),
    );
  }
}

class HomeScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Home')),
      body: Center(child: Text('Welcome to the App!')),
    );
  }
}

void main() {
  runApp(MaterialApp(
    home: SplashScreen(),
  ));
}
