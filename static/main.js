$(document).ready(function () {
    var open = 1;

    //expansion 
    $('#gopub').on("click", function (event) {

        if (open == 1) {
            open = 0;
            $("#addme").slideUp("slow");
        }
        else {
            open = 1;
            $("#addme").slideDown("slow");
        }
    });




});