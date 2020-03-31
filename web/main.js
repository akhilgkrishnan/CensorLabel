$(function(){
    $("#btn_add").click(function(){
        if($('#inp').val()!='')
          {
            eel.handleinput($("#inp").val());
          }
        else
        {
            info("name cant be NULL");
        }  
        $('#inp').val('');
    });

    $("#btn_detect").click(function()
    {
        eel.detect_faces();

    })

    $("#btn_train").click(function()
    {
        eel.train_images();

    })




});

eel.expose(detected_name);
function detected_name(x) {
    console.log("Hello from " + x);
    document.getElementById('uName').innerHTML='Detected person is :'+x;

}

eel.expose(info);
function info(x) {
        document.getElementById('uName').innerHTML=x;
    }

eel.expose(mSpinner);       
function mSpinner() {
    v   ar x = document.getElementById("mSpinner");
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

