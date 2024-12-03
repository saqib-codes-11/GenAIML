//variables to queue synthesised sounds in the browser
var k = -1;
var sounds = [];

//session id for unique temp files on server
sessionId = Date.now();
console.log(sessionId);

//variables for keyword generation
var useOwnKW = false;

//default keyword settings
var KW = {
  model: "laurie",
  keywords: "blue ball",
};

//Queue object to store text generation
var q = new Queue();

var textBox = document.getElementById("textBox");
console.log(textBox);

//FUNCTION
//get a random int between min and max
function randomIntFromInterval(min, max) {
  // min and max included
  return Math.floor(Math.random() * (max - min + 1) + min);
}

//are we in automated mode - to automatically produce output
var automated = false;
var addingToQueue = false;

//function to start a new output if directed from automation
function automatedGo() {
  console.log("Start of an automated text and audio generation");
  //randomly select a keyword generation source
  kwGenerator = randomIntFromInterval(1, 6);
  console.log("Keyword Source");
  switch (kwGenerator) {
    case 1:
      console.log("Google Aus");
      clickGoogleAus();

      break;
    case 2:
      console.log("Google USA");
      clickGoogleUSA();

      break;
    case 3:
      console.log("Reddit News Hot");
      clickRedditNewsHot();

      break;
    case 4:
      console.log("Reddit News - New");
      clickRedditNewsNew();

      break;
    case 5:
      console.log("Reddit Funny - Hot");
      clickRedditFunnyHot();

      break;
    case 6:
      console.log("Reddit Funny - New");
      clickRedditFunnyNew();

      break;

    default:
      clickGoogleAus();
      break;
  }
}

function onclickStart() {
  if (!automated) {
    startButton = document.getElementById("startauto");
    startButton.style.backgroundColor = "green";
    disableGoButton();
    disableDiv(document.getElementById("controlsLeft"));
    automated = true;
    automatedGo();
  }
}

//FUNCTION - user clicks the Stop Automation button
//my browser seems to hold on to some state information on refresh - so changing the automation variable and refrehshing
//the refresh will stop the audio playing - if you remove the refresh it will keep playing but just not start another generation cycle
function onclickStop() {
  automated = false;
  location.reload();
}

//FUNCTION - user clicks the Go button
//this is also called in general to start round of generation - so is also involved in automation
//automation has a queue to store generated text - so this function only calls text generation if the queue is empty
function onLoad() {
  textBox.innerHTML = "Generating........";

  disableDiv(document.getElementById("controlsLeft"));
  if (!automated) disableDiv(document.getElementById("controlsRight"));
  var xhttp = new XMLHttpRequest();

  //if using an online source of keywords they will already be set
  //if user has entered their own keywords then we update the UI
  keywordEntry = document.getElementById("kwToUse");
  if (useOwnKW) KW.keywords = keywordEntry.value;

  //on XHR response
  xhttp.onreadystatechange = function () {
    //on SUCCESS we send the text off for synthesis
    if (this.readyState == 4 && this.status == 200) {
      let r = JSON.parse(this.responseText);

      let output = "";
      output = r.replace('"', "");
      output = output.replace(/\n/g, "<br>");

      //store the keyqords used nd the generated text and add them to the generated text queue
      kWTextPair = {
        keywords: KW.keywords,
        qText: r,
      };

      pairToSynth = kWTextPair;
      document.getElementById("keyWords").innerHTML = pairToSynth["keywords"];

      text = { text: pairToSynth.qText };

      console.log("Sending for synthesis:");
      console.log(text["text"]);
      console.log("------------");
      synth(text);
    }
    //on SERVER ERROR WITH TEXT GENERATION
    //update user with ERROR - display the keywords it failed on
    //
    if (this.readyState == 4 && this.status == 500) {
      textBox.innerHTML =
        "ERROR WITH TEXT GENERATION<br>" +
        "Keywords used:<br>" +
        KW["keywords"];
      //if automated change the button colour to red and stop automation
      if (automated) {
        startButton = document.getElementById("startauto");
        startButton.style.backgroundColor = "red";
        automated = false;
      }
    }
  };
  //if the queue is empty then send the request
  if (q.isEmpty) {
    xhttp.open("POST", "http://34.122.148.252:5000/generate", true);
    xhttp.setRequestHeader("Content-Type", "application/json");
    xhttp.send(JSON.stringify(KW));
  }
  //otherwise take text off the queue and send for synthesis
  else {
    pairToSynth = q.dequeue();
    document.getElementById("keyWords").innerHTML = pairToSynth["keywords"];

    let output = "";
    output = pairToSynth.qText.replace('"', "");
    output = output.replace(/\n/g, "<br>");

    text = { text: pairToSynth["qText"] };
    synth(text);
  }
}

//FUNCTION playSounds()
//INPUT: array containing audio elements of synthesised speech, integer (k) to keep track of which line to play/play next
function playSounds() {
  k++;
  //if we have played the last sound
  if (k == sounds.length) {
    //if we are automated then start another round of text generation and voice synthesis
    if (automated) {
      //initially had text fade IN - this would reset to an opacity of 0 to fade back in
      //textBox = document.getElementById("textBox");
      //textBox.classList.remove("fade-in-text");
      //textBox.style.opacity = 0;

      automatedGo();
    }
    //otherwise unlock the user interface
    else {
      reenableDiv(document.getElementById("controlsLeft"));
      reenableDiv(document.getElementById("controlsRight"));
    }

    sounds = [];
    return;
  }

  //listen for when this sound ends and then going to play the next sound
  sounds[k].addEventListener("ended", playSounds);

  //play this sound
  sounds[k].play();
}

//FUNCTION to make a ID for file request - browser was caching them without a unique request
//INPUT: integer length of unique ID
//OUTPUT: id which can consist of letters and digits
function makeid(length) {
  var result = "";
  var characters =
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
  var charactersLength = characters.length;
  for (var i = 0; i < length; i++) {
    result += characters.charAt(Math.floor(Math.random() * charactersLength));
  }
  return result;
}

//FUNCTION to synthesis speech from text
//input the text from the text geneartion
//output - the files that the server generated are played to the user

function synth(text) {
  console.log("Sending speech synthesis request");
  //when synthesis starts a new worker is generated to add text generation to the queue of text
  var worker = new Worker("static/worker.js");

  _text = text["text"];
  //get rid of any quotes
  output = _text.replace('"', "");
  //change the new line character to html for display to user
  output = output.replace(/\n/g, "<br>");

  //split original text into lines to count them
  tList = _text.split("\n");
  numLines = tList.length;

  //we will send an object with the current user sessionId and the text to synthesise
  uText = { session: sessionId, text: text };

  var xhttp = new XMLHttpRequest();

  //audio = document.getElementById("audio");

  xhttp.onreadystatechange = function () {
    //on success add all the generated files to an array and call playSounds() to play to user
    if (this.readyState == 4 && this.status == 200) {
      r = JSON.parse(this.responseText);

      textBox.innerHTML = output;

      console.log("Speech Synthesised OK\n---");
      sounds = new Array();

      numLines = r["numLines"];
      for (let i = 0; i < numLines - 1; i++) {
        // console.log("Adding sound " + (i + 1));
        ranId = makeid(5);
        newSound = new Audio(
          "static/output/" + sessionId + "/line" + i + ".wav?" + ranId
        );
        newSound.load();
        sounds.push(newSound);
      }
      //let textBox = document.getElementById("textBox");
      //when text would fade in this triggered it to fade in
      //textBox.classList.add("fade-in-text");
      k = -1;
      //if not automated we aren't going to add more text to the text generatin queue
      if (!automated) {
        playSounds(...sounds, k);
      }
      //if automated then create a jscript worker to request more text for the text queue
      if (automated) {
        if (q.length < 2) {
          worker.addEventListener("message", function (e) {
            q.enqueue(e.data);
            worker.terminate();
          });

          worker.postMessage("GO");
        }
        playSounds(...sounds, k);
      }
    }
    // on failure - usually failure is that the voice model decorder ran out of steps
    //report to the user the line it failed on and the entire text that was sent
    if (this.readyState == 4 && this.status == 500) {
      r = JSON.parse(this.responseText);
      textBox.innerHTML =
        "ERROR<br><br>Audio Synthesis Failed on Line:<br><br>" +
        r["line"] +
        "<br><br>-----</br>" +
        output +
        "-----";
      //if automated set UI colour to red to signal problem
      if (automated) {
        startButton = document.getElementById("startauto");
        startButton.style.backgroundColor = "red";
        automated = false;
      }
      //unlock the UI as an error has occured
      reenableDiv(document.getElementById("controlsLeft"));
      reenableDiv(document.getElementById("controlsRight"));
    }
  };
  xhttp.open("POST", "http://34.122.148.252:5000/synthesise", true);
  xhttp.setRequestHeader("Content-Type", "application/json");
  xhttp.send(JSON.stringify(uText));
}

//CSS is used to disable clicking inside a DIV
//INPUT: DOM div
//output: the div class is changed
function disableDiv(div) {
  div.classList.add("divDisabled");
}

//CSS is used to enable clicking inside a DIV
//INPUT: DOM div
//output: the div class is changed
function reenableDiv(div) {
  div.classList.remove("divDisabled");
}

//functions to hide elements of UI

//FUNCTION to hide controls
//Input: none
//Outut: controls are hidden or displayed depending on current state
controlsHidden = false;
function hideControls() {
  controls = document.getElementById("controls");
  if (controlsHidden == false) {
    controls.style.display = "none";
    controlsHidden = true;
  } else {
    controls.style.display = "flex";
    controlsHidden = false;
  }
}

//FUNCTION to hide keywords
//Input: none
//Outut: controls are hidden or displayed depending on current state
keywordsHidden = false;
function hideKeywords() {
  keywords = document.getElementById("keywordsContainer");
  if (keywordsHidden == false) {
    keywords.style.display = "none";
    keywordsHidden = true;
  } else {
    keywords.style.display = "inline";
    keywordsHidden = false;
  }
}

//function to enable the GO button
function enableGoButton() {
  button = document.getElementById("goButton");
  button.disabled = false;
}

//function to disable the GO butto
function disableGoButton() {
  button = document.getElementById("goButton");
  button.disabled = true;
}

//FUNCTION to get keywords from Google Australia
//INPUT:none
//OUTPUT: keywords from Google Trends using pyTrends
function clickGoogleAus() {
  disableGoButton();

  document.getElementById("keyWords").innerHTML = "";
  dropButton = document.getElementById("dropButton");
  dropButton.innerHTML = "Trending Google Searches - Australia";

  input = document.getElementById("kwToUse");
  input.disabled = true;

  useOwnKW = false;

  var xhttp = new XMLHttpRequest();

  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      KW.keywords = this.responseText;

      document.getElementById("keyWords").innerHTML = KW["keywords"];
      if (automated) if (!addingToQueue) onLoad();
      if (!automated) enableGoButton();
    }
    if (this.readyState == 4 && this.status == 500) {
      document.getElementById("keyWords").innerHTML =
        "FAILED TO RETRIEVE KEYWORDS";
    }
  };

  xhttp.open("GET", "http://34.122.148.252:5000/kwGoogleAus", true);
  xhttp.send(null);
}

//FUNCTION to get keywords from Google USA
//INPUT:none
//OUTPUT: keywords from Google Trends using pyTrends
function clickGoogleUSA() {
  disableGoButton();

  document.getElementById("keyWords").innerHTML = "";
  dropButton = document.getElementById("dropButton");
  dropButton.innerHTML = "Trending Google Searches - USA";

  input = document.getElementById("kwToUse");
  input.disabled = true;

  useOwnKW = false;

  var xhttp = new XMLHttpRequest();

  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      //let r = JSON.parse(this.responseText);
      KW.keywords = this.responseText;
      //console.log(r)

      document.getElementById("keyWords").innerHTML = KW["keywords"];
      if (automated) if (!addingToQueue) onLoad();
      if (!automated) enableGoButton();
    }
    if (this.readyState == 4 && this.status == 500) {
      document.getElementById("keyWords").innerHTML =
        "FAILED TO RETRIEVE KEYWORDS";
    }
  };

  xhttp.open("GET", "http://34.122.148.252:5000/kwGoogleUSA", true);
  xhttp.send(null);
}

//FUNCTION to get keywords from REDDIT - subreddit News (Hot posts)
//INPUT:none
//OUTPUT: keywords with stop words removed
function clickRedditNewsHot() {
  disableGoButton();

  document.getElementById("keyWords").innerHTML = "";
  dropButton = document.getElementById("dropButton");
  dropButton.innerHTML = "Reddit - Subreddit:News - Hot";

  input = document.getElementById("kwToUse");
  input.disabled = true;

  useOwnKW = false;

  var xhttp = new XMLHttpRequest();

  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      KW.keywords = this.responseText;

      document.getElementById("keyWords").innerHTML = KW["keywords"];
      if (automated) if (!addingToQueue) onLoad();
      if (!automated) enableGoButton();
    }
    if (this.readyState == 4 && this.status == 500) {
      document.getElementById("keyWords").innerHTML =
        "FAILED TO RETRIEVE KEYWORDS";
    }
  };

  xhttp.open("GET", "http://34.122.148.252:5000/kwRedditNewsHot", true);
  xhttp.send(null);
}
//FUNCTION to get keywords from REDDIT - subreddit News (New posts)
//INPUT:none
//OUTPUT: keywords with stop words removed
function clickRedditNewsNew() {
  disableGoButton();
  document.getElementById("keyWords").innerHTML = "";
  dropButton = document.getElementById("dropButton");
  dropButton.innerHTML = "Reddit - Subreddit:News - New";

  input = document.getElementById("kwToUse");
  input.disabled = true;

  useOwnKW = false;

  var xhttp = new XMLHttpRequest();

  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      //let r = JSON.parse(this.responseText);
      KW.keywords = this.responseText;
      //console.log(r)

      document.getElementById("keyWords").innerHTML = KW["keywords"];
      if (automated) if (!addingToQueue) onLoad();
      if (!automated) enableGoButton();
    }
    if (this.readyState == 4 && this.status == 500) {
      document.getElementById("keyWords").innerHTML =
        "FAILED TO RETRIEVE KEYWORDS";
    }
  };

  xhttp.open("GET", "http://34.122.148.252:5000/kwRedditNewsNew", true);
  xhttp.send(null);
}
//FUNCTION to get keywords from REDDIT - subreddit Funny (New posts)
//INPUT:none
//OUTPUT: keywords with stop words removed
function clickRedditFunnyNew() {
  disableGoButton();
  document.getElementById("keyWords").innerHTML = "";
  dropButton = document.getElementById("dropButton");
  dropButton.innerHTML = "Reddit - Subreddit:Funny - New";

  input = document.getElementById("kwToUse");
  input.disabled = true;

  useOwnKW = false;

  var xhttp = new XMLHttpRequest();

  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      //let r = JSON.parse(this.responseText);
      KW.keywords = this.responseText;
      //console.log(r)

      document.getElementById("keyWords").innerHTML = KW["keywords"];
      if (automated) if (!addingToQueue) onLoad();
      if (!automated) enableGoButton();
    }
    if (this.readyState == 4 && this.status == 500) {
      document.getElementById("keyWords").innerHTML =
        "FAILED TO RETRIEVE KEYWORDS";
    }
  };

  xhttp.open("GET", "http://34.122.148.252:5000/kwRedditFunnyNew", true);
  xhttp.send(null);
}
//FUNCTION to get keywords from REDDIT - subreddit Funny (Hot posts)
//INPUT:none
//OUTPUT: keywords with stop words removed
function clickRedditFunnyHot() {
  disableGoButton();
  document.getElementById("keyWords").innerHTML = "";
  dropButton = document.getElementById("dropButton");
  dropButton.innerHTML = "Reddit - Subreddit:Funny - Hot";

  input = document.getElementById("kwToUse");
  input.disabled = true;

  useOwnKW = false;

  var xhttp = new XMLHttpRequest();

  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      //let r = JSON.parse(this.responseText);
      KW.keywords = this.responseText;
      //console.log(r)

      document.getElementById("keyWords").innerHTML = KW["keywords"];
      if (automated) if (!addingToQueue) onLoad();
      if (!automated) enableGoButton();
    }
    if (this.readyState == 4 && this.status == 500) {
      document.getElementById("keyWords").innerHTML =
        "FAILED TO RETRIEVE KEYWORDS";
    }
  };

  xhttp.open("GET", "http://34.122.148.252:5000/kwRedditFunnyHot", true);
  xhttp.send(null);
}

//FUNCTION to allow user to enter own keywords for generation
//INPUT: none
//OUTPUT: enables the input field
function clickEnterYourOwn() {
  dropButton = document.getElementById("dropButton");
  dropButton.innerHTML = "Enter Your Own Keywords";

  input = document.getElementById("kwToUse");
  input.disabled = false;

  //when we generate text we know that they haven't been set yet by the functions that source keywords online
  useOwnKW = true;
}

//FUNCTION to update the Keywords UI display when user keywords are entered and enable the go button
function updateKWTBU() {
  input = document.getElementById("kwToUse");
  document.getElementById("keyWords").innerHTML = input.value;
  enableGoButton();
}
