/* Sumbit functionality for annotation task */

$("#commit").click(function (ev) {
    /** list checkbox values **/
    var num_images_annotated = checkAllFieldsets();
    console.log(num_images_annotated);
    if (num_images_annotated > -1) {
	alert('Please annotate image '+(num_images_annotated+1).toString()+' before submitting.');
	return;
    }
    var checks = $('.opt-check').map(
	function () { 
	   inds = this.id.split('_');
	   image_id = imgs[inds[0]]; 
	   label_id = label_ids[inds[1]]; 
	   patch_id = patch_ids[inds[0]]; 
	   return {image_id:image_id, 
	   	label_id:label_id, 
	   	patch_id:patch_id, 
	   	value:this.checked};
	}).toArray();

    $("#resp").val(JSON.stringify(checks));

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
    $.ajax({
        type: "POST",
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
		console.log('website submit!');
                console.dir($(that).parent().serialize());
		// TODO: fix to account for allimgs hits
		window.location.replace("/task/"+String(parseInt(window.location.href.split('/')[window.location.href.split('/').length-2])+1));
		// location.reload();
            }
        },
        error: function()
        {
            //if workerId, submit to mturk
            //else simply reload page
	    console.log("There was a server error, however this form will submit to mturk.")
            if(worker.lastIndexOf('mturk_', 0) === 0) {
                console.dir($(that).parent().serialize());
                console.dir($("#mturk").serialize());
                $(that).parent().submit();
            }
            else {
		console.log('website submit!');
                console.dir($(that).parent().serialize());
		// TODO: fix to account for allimgs hits
		window.location.replace("/task/"+String(parseInt(window.location.href.split('/')[window.location.href.split('/').length-2])+1));
		// location.reload();
            }
        },
        statusCode: {
          500: function() {
            //if workerId, submit to mturk
            //else simply reload page
	    console.log("There was a 500 server error, however this form will submit to mturk.")
            if(worker.lastIndexOf('mturk_', 0) === 0) {
                console.dir($(that).parent().serialize());
                console.dir($("#mturk").serialize());
                $(that).parent().submit();
            }
            else {
		console.log('website submit!');
                console.dir($(that).parent().serialize());
		// TODO: fix to account for allimgs hits
		window.location.replace("/task/"+String(parseInt(window.location.href.split('/')[window.location.href.split('/').length-2])+1));
		// location.reload();
            }
           }
        },
    });

});

/* Function to fill in hit responses for replay */
function replay_annotations( annotations ){
    for (patch_ind = 0; patch_ind < annotations.length; patch_ind++) {

	for (attr_ind = 0; attr_ind < annotations[patch_ind].length; attr_ind++) {
	    if (annotations[patch_ind][attr_ind]) {
		$('#'+patch_ind.toString()+'_'+attr_ind.toString()).prop('checked', true);
	    } else {
		$('#'+patch_ind.toString()+'_1').prop('checked', true);
	    }
	}
    }

}
