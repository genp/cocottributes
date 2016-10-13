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

/* Task page functions */


$(window).load(function() {

    loadPage();
    $( document ).tooltip();

});

function checkAllFieldsets(){
    var all_checked = $('fieldset').map(
        function () { 
    	return $(this).find("input:checked").length >=1 ? 1 : 0;
        }).toArray()
    return all_checked.indexOf(0);
}

$('.none-check').change(
    function(){
	$(this).closest("fieldset").find('.opt-check').each(
	    function () { 
		this.checked = false; 
	    });
	var cid = $(this).prop("id").split('_')[0];
	if ($('#c'+cid).parent().prop("class") == "clickablebox") {
	    $('#c'+cid).removeClass('positive');
	    }	
    });

$('.opt-check').change(
    function() {
	$(this).closest('fieldset').find('.none-check').attr('checked', false); 
	var cid = $(this).prop("id").split('_')[0];

	if ($('#c'+cid).parent().prop("class") == "clickablebox") {
	    $('#c'+cid).toggleClass('positive');
	    }
    });
