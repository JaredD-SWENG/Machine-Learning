function getResult() {

    var url = "http://localhost:8000";   // The URL and the port number must match server-side
    var endpoint = "/result";            // Endpoint must match server endpoint

    var http = new XMLHttpRequest();

    // prepare GET request
    http.open("GET", url + endpoint, true);

    http.onreadystatechange = function() {
        var DONE = 4;       // 4 means the request is done.
        var OK = 200;       // 200 means a successful return.
        if (http.readyState == DONE && http.status == OK && http.responseText) {

            // JSON string
            var replyString = http.responseText;

            var resultData = JSON.parse(replyString); // Parse JSON string into JavaScript object

            // Display the accuracy, precision, and recall on separate lines
            document.getElementById("result").innerHTML = "Accuracy: " + resultData.accuracy + "<br>";
            document.getElementById("result").innerHTML += "Precision: " + resultData.precision + "<br>";
            document.getElementById("result").innerHTML += "Recall: " + resultData.recall + "<br>";

        }
    };

    // Send request
    http.send();
}
