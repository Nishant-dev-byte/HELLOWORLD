import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

class Login extends StatefulWidget {
  const Login({Key? key}) : super(key: key);

  @override
  _LoginState createState() => _LoginState();
}

class _LoginState extends State<Login> {

  TextEditingController user = TextEditingController();
  TextEditingController password = TextEditingController();
  
  SignIn()async{
    await FirebaseAuth.instance.signInWithEmailAndPassword(
      email: user.text,
      password: password.text
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Login Page'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          children: [
            TextField(
              controller: user,
              decoration: const InputDecoration(
                hintText: 'User',
            ),
          ),
          TextField(
            controller: password,
            decoration: const InputDecoration(
              hintText: 'Password',
            ),
          ),
          ElevatedButton(
            onPressed: (()=>SignIn()),
            child: const Text('Login'),
          ),
        ],
        ),
      ),
    );
  }
}