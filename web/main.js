
function start(){
        var video_path = document.getElementById("video-path").value
        var movie_lang = document.getElementById("movie-lang").value
        var gpu_support = document.getElementById("gpu-support").value
        var display_frame = document.getElementById("display-frame").value
        document.getElementById('pname').innerHTML=video_path;
        eel.startLabel(video_path,movie_lang,gpu_support,display_frame)

}



eel.expose(info);
function info(x) {
        document.getElementById('uName').innerHTML=x;
    }

eel.expose(mSpinner);       
function mSpinner() {
    var x = document.getElementById("mSpinner");
    if (x.style.display === "block") {
        x.style.display = "none";
} else {
    x.style.display = "block";
}
}

eel.expose(mAddTick);
function mAddTick()
{
var x = document.getElementById("mAddTick");
if (x.style.display === "block") {
    x.style.display = "none";
} else {
    x.style.display = "block";
}                    }

