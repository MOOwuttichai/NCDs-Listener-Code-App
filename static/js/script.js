function createItem_bd() {
  localStorage.setItem("set", 1);
}

function createItem_ad() {
  localStorage.setItem("set", 2);
}

function readValue_bd() {
  var x = localStorage.getItem("set");
  if (x = 1) {
    document.getElementById("myBtn1").disabled = false;
    } else if (x = 2) {
    document.getElementById("myBtn1").disabled = false;
    document.getElementById("myBtn2").disabled = false;
    } else {
      document.getElementById("myBtn1").disabled = true;
      document.getElementById("myBtn2").disabled = true;
    }
}