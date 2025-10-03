import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;

class Homepage extends StatefulWidget {
  const Homepage({Key? key}) : super(key: key);

  @override
  State<Homepage> createState() => _HomepageState();
}

class _HomepageState extends State<Homepage> {
  final User? user = FirebaseAuth.instance.currentUser;
  String? selectedFileName;
  String? selectedVideoName;

  Future<void> signout() async {
    await FirebaseAuth.instance.signOut();
  }

  Future<void> pickImage() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      allowMultiple: false,
      withData: true, // Important for web to get file data
    );

    if (result != null && result.files.isNotEmpty) {
      setState(() {
        selectedFileName = result.files.first.name;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Photo selected: $selectedFileName')),
      );
    }
  }

  Future<void> pickVideo() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.video,
      allowMultiple: false,
      withData: true, // Important for web to get file data
    );

    if (result != null && result.files.isNotEmpty) {
      setState(() {
        selectedVideoName = result.files.first.name;
      });
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Video selected: $selectedVideoName')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey[100], // Add background color
      appBar: AppBar(
        backgroundColor: Colors.deepPurple, // Theme color
        title: const Text('Web Upload', style: TextStyle(color: Colors.white)),
        elevation: 4, // Add shadow
        actions: [
          IconButton(
            icon: const Icon(Icons.logout, color: Colors.white),
            onPressed: signout,
            tooltip: 'Sign Out',
          ),
        ],
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Colors.grey[100]!, Colors.grey[200]!],
          ),
        ),
        child: Center(
          child: Card(
            margin: const EdgeInsets.all(16),
            elevation: 8,
            child: Padding(
              padding: const EdgeInsets.all(24),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton.icon(
                    onPressed: pickImage,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.deepPurple,
                      padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                    ),
                    icon: const Icon(Icons.image, color: Colors.white),
                    label: const Text('Select Image', style: TextStyle(color: Colors.white)),
                  ),
                  if (selectedFileName != null)
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Text(
                        'Selected image: $selectedFileName',
                        style: const TextStyle(fontSize: 14, color: Colors.black87),
                      ),
                    ),
                  const SizedBox(height: 24),
                  ElevatedButton.icon(
                    onPressed: pickVideo,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.deepPurple,
                      padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 16),
                    ),
                    icon: const Icon(Icons.video_library, color: Colors.white),
                    label: const Text('Select Video', style: TextStyle(color: Colors.white)),
                  ),
                  if (selectedVideoName != null)
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Text(
                        'Selected video: $selectedVideoName',
                        style: const TextStyle(fontSize: 14, color: Colors.black87),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}