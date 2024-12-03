# AIML-art Project "Voice of Laurie Anderson"
Web app to generate text and audio in the style and voice of Laurie Anderson and display/play this to the user.

![lyricgenTTS Homepage](https://github.com/Sturok/aimlartLyricGenTTS/blob/main/images/LyricgenTTS.png)

## Installation
# This project is setup to run the Gunicorn WSGI server in a Linux environment
1. Clone this repository
2. Install dependencies "pip install -r requirements.txt"
4. Download 'LaurieAudiobook117000' from the Google Bucket and place in the root directory of this project
5. Download contents of directory Hifigan from Google Bucket and place in the /Hifigan directory of this project
6. Download Lyricgen text generations model(s) from Google Bucket and place in the /models directory of this project (currently set to use the 'Laurie' model)
7. Edit the XHR request IP addresses in **/static/tts.js** and **/static/worker.js** to the IP address the server will run on - currently set to **34.122.148.252:5000** which is the location of the Google Computer   Engine VM at the time of writing this. 
Change to something like **127.0.0.1:5000**

## Run the server
Basic options for Gunicorn:
-w *number of workers (Default sync worker type)*

-t *worker timeout*

-b *address and port to bind to*

To run the automated generation function multiple workers are required to avoid blocking.

**gunicorn main:app -w 4 -t 480 -b 0.0.0.0:5000**
The webapp is coded to send its XHR requests to port 5000

### Start the cleanup app to remove the output files
**gunicorn cleanup:app -b 0.0.0.0:8000**
The port of the cleanup app just needs to be different to 5000

## Access the web app
Access *http://server-address:5000/* to be delivered the tts.html page from the server
