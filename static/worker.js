var KW = {
  model: "laurie",
  keywords: "DEFAULT KEYWORDS",
};

//get a random int between min and max
function randomIntFromInterval(min, max) {
  // min and max included
  return Math.floor(Math.random() * (max - min + 1) + min);
}

function automatedGenerate() {
  var xhttp = new XMLHttpRequest();
  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      let r = JSON.parse(this.responseText);

      let output = r.replace('"', "");
      output = output.replace(/\n/g, "<br>");
      //console.log(output)
      pairToSend = { keywords: KW.keywords, qText: r };
      self.postMessage(pairToSend);
      console.log("Sent back response. Killing worker.");
      //self.close();
    }
  };

  xhttp.open("POST", "http://34.122.148.252:5000/generate", true);
  xhttp.setRequestHeader("Content-Type", "application/json");
  xhttp.send(JSON.stringify(KW));
}

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

function clickGoogleAus() {
  var xhttp = new XMLHttpRequest();

  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      //let r = JSON.parse(this.responseText);
      KW.keywords = this.responseText;
      automatedGenerate();
      //console.log(r)
    }
  };

  xhttp.open("GET", "http://34.122.148.252:5000/kwGoogleAus", true);
  xhttp.send(null);
}

function clickGoogleUSA() {
  var xhttp = new XMLHttpRequest();

  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      //let r = JSON.parse(this.responseText);
      KW.keywords = this.responseText;
      automatedGenerate();
      //console.log(r)
    }
  };

  xhttp.open("GET", "http://34.122.148.252:5000/kwGoogleUSA", true);
  xhttp.send(null);
}

function clickRedditNewsHot() {
  var xhttp = new XMLHttpRequest();

  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      //let r = JSON.parse(this.responseText);
      KW.keywords = this.responseText;
      automatedGenerate();
      //console.log(r)
    }
  };

  xhttp.open("GET", "http://34.122.148.252:5000/kwRedditNewsHot", true);
  xhttp.send(null);
}
function clickRedditNewsNew() {
  var xhttp = new XMLHttpRequest();

  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      //let r = JSON.parse(this.responseText);
      KW.keywords = this.responseText;
      automatedGenerate();
      //console.log(r)
    }
  };

  xhttp.open("GET", "http://34.122.148.252:5000/kwRedditNewsNew", true);
  xhttp.send(null);
}

function clickRedditFunnyNew() {
  var xhttp = new XMLHttpRequest();

  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      //let r = JSON.parse(this.responseText);
      KW.keywords = this.responseText;
      automatedGenerate();
      //console.log(r)
    }
  };

  xhttp.open("GET", "http://34.122.148.252:5000/kwRedditFunnyNew", true);
  xhttp.send(null);
}

function clickRedditFunnyHot() {
  var xhttp = new XMLHttpRequest();

  xhttp.onreadystatechange = function () {
    if (this.readyState == 4 && this.status == 200) {
      //let r = JSON.parse(this.responseText);
      KW.keywords = this.responseText;
      automatedGenerate();
      //console.log(r)
    }
  };

  xhttp.open("GET", "http://34.122.148.252:5000/kwRedditFunnyHot", true);
  xhttp.send(null);
}

self.addEventListener("message", function (e) {
  console.log("Adding to queue as separate worker");

  addingToQueue = true;
  console.log("Worker getting keywords");
  automatedGo();
  console.log("Worker received keywords");
});
