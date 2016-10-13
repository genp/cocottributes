/* Sumbit button functionality for quiz task */
$("#content").append('<div id="score"  class="sticky" style="display:inline-block"></div>');
$("#score").hide();

function displayScore( score ) { 
    var msg = '';
    if ( score > 0.8 ) {
	msg = "<b>You passed!</b> Please scroll to the top to see any attributes you may have missed. They are highlighted in red. Attributes that could have been either true or false are highlighted in blue. Now you can do thousands of "+task_type+" Annotation HITs. Good luck!";
    } else {
	msg = "<b>You did not pass.</b> Please scroll to the top to see which attributes you missed. Incorrect selections are highlighted in red. Attributes that could have been either true or false are highlighted in blue. Try to take this quiz again. When you pass you will be able to do all of  the  "+task_type+" Annotation HITs. Good Luck!";
    }
    
    $("#score").html(msg);
    $("#score").show();
    
}

function sumTuple(a, b) { 
    res = [];
    for (var i=0; i < a.length; i++ ) {
	res.push(a[i]+b[i]);
    }
    return res; 
}

$("#commit").click(function (ev) {
    /** list checkbox values **/
    var num_images_annotated = checkAllFieldsets();
    console.log(num_images_annotated);
    if (num_images_annotated > -1) {
	alert('Please annotate image '+(num_images_annotated+1).toString()+' before submitting.');
	return;
    }
    var score = $('.opt-check').map(
	function () { 
	    inds = this.id.split('_');
	    patch_id = patch_ids[inds[0]];
	    label_id = label_ids[inds[1]];
	    var tp_fp_tn_fn = [0, 0, 0, 0]; 
	    // is_correct = ans[patch_id.toString()][label_id.toString()] == this.checked ? 1 : 0;
	    var gt = ans[patch_id.toString()][label_id.toString()];
	    var alt_gt = alt_ans[patch_id.toString()][label_id.toString()];
	    if ( gt == 1 & this.checked) {
		tp_fp_tn_fn[0] = 1;
	    } else if ( gt == 1 & !this.checked ) {
		tp_fp_tn_fn[1] = 1;
		if (alt_gt == 0) {
		    $(this).next('label').css({ 'color': 'blue'});
		} else {
		    $(this).next('label').css({ 'color': 'red'});
		}
	    } else if ( gt == 0 & !this.checked ) {
		if (alt_gt == 1) {
		    $(this).next('label').css({ 'color': 'blue'});
		}
		tp_fp_tn_fn[2] = 1;
	    } else if ( gt == 0 & this.checked ) {
		tp_fp_tn_fn[3] = 1;
		if (alt_gt == 1) {
		    $(this).next('label').css({ 'color': 'blue'});
		} else {
		    $(this).next('label').css({ 'color': 'red'});
		}
	    }
	    return [tp_fp_tn_fn];
	}).toArray();
    
    var sum = score.reduce(sumTuple);
    var avg = [];
    for (var i=0; i < 2; i++ ) {
	avg.push(sum[i] / num_pos);
    }
    for (var i=2; i < 4; i++ ) {
	avg.push(sum[i] / num_neg);
    }

    displayScore(avg[0]);
    $("#resp").val(JSON.stringify(avg));

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
    $("#assignment_id").val(assignmentId);
    $("#hit_id").val(hitId);

    var data = $("#submit").serialize();
    var that = this;
    setTimeout( function () {
    $.ajax({
        type: "POST",
        data: data,
        success: function()
        {
            //if workerId, submit to mturk

            if(worker.lastIndexOf('mturk_', 0) === 0) {
                $(that).parent().submit();
            }
            else {
		console.log('website submit!');
		location.reload();
            }
        }
    });
    }, avg[0] < 0.8 ? 60000 : 30000);

});

