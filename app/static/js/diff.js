/*
Code for handling worker data
*/

var start = 0;
var assignmentId = '';
var workerId = '';
var hitId = '';
var worker = '';
var confidence = 0;
var ip = 0;
var nationality = '';

/*
Not using nav bar because this is a HIT
*/

$("#nav_bar").remove()
$("#content").css('margin-top', '0')

$(window).load(function() {
 loadPage()
});

/*
Adding bullet list of previous attributes
*/

function list_words() {
    $('#cur_words').append("<ul id='words_list'></ul>");
    for (cnt = 0; cnt < cur_words.length; cnt++) {
        $("#words_list").append("<li>"+cur_words[cnt]+"</li>");
    }
    if (cur_words.length === 0) {
	$("#words_list").append("<li> Nobody thought of any good ones yet! </li>");
    }
}

list_words();
$('#submit_button').attr('disabled', true);

$("#commit").click(function (ev) {
    /** this returns a time in seconds*/
    time = (new Date - start) / 1000;

    /** Prevent form submission */
    ev.preventDefault();
    $('#title').html("<h1> Form submiting..... </h1>");
    $(this).prop("disabled",true);

    if (workerId) {
        $("#worker").val(worker);
    } else {
        $("#worker").val($('#worker_text').val());
    }
    $("#time").val(time);
    $("#location").val(ip);
    $("#nationality").val(nationality);


    $('.text').change();
    $('#resp').val(JSON.stringify(user_input));
    console.log($('#resp'));

    var data = $("#submit_diff").serialize();
    var that = this;
    $.ajax({
        type: "POST",
        //url: "/diff/"+$("#id").val(),
        data: data,
        success: function()
        {
            //if workerId, submit to mturk
            //else simply reload page

            if(worker.lastIndexOf('mturk_', 0) === 0) {
                console.dir($(that).parent().serialize());
                console.dir($("#mturk").serialize());
                $(that).parent().submit();
            }
            else {
		console.log('else');
                console.dir($(that).parent().serialize());
                console.dir($("#mturk").serialize());
		location.reload();
            }
        }
    });

});


$('.text').change(function() {
  $('input.text').each(function( index ) {
    var word = $(this).val().trim();
    if ( word != "" && user_input.indexOf(word) === -1 && cur_words.indexOf(word) === -1 ){
      user_input.push(word);
    }
  });
  console.log(user_input);
  if ( user_input.length >= 3 ){
    $('#commit').removeAttr('disabled');
  }

});
