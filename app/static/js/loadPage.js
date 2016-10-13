function loadPage() {
    start = new Date;
    get_location();

    /* Get query parameters */
    var queryParameters = getUrlVars();

    /* Given an assignmentId, add to the form. */
    assignmentId = queryParameters.assignmentId;

    if(assignmentId) {
      $("#mturk").append($('<input/>').attr('type', 'hidden').attr('name', 'assignmentId').val(assignmentId));
    } else {
	assignmentId = 'none';
    }

    /* Given an workerId, add to the form. */
    workerId = queryParameters.workerId;

    if (parent != window) {
	if (document.referrer.indexOf('mturk') > -1) {
            console.log(workerId);
	    if (workerId == null) {
		console.log('no worker');
		$('#commit').prop("disabled", true);
		$('#commit').html("Please accept this HIT before working on it.");
	    } else {
		worker = "mturk_"+workerId;
		console.log(worker);
		$("#mturk").append($('<input/>').attr('type', 'hidden').attr('name', 'workerId').val(workerId));
	    }
	}
    } else {

        // put in worker textarea
        $("#input-area").append($('<input/>').attr('type', 'text').attr('id', 'worker_text').attr('placeholder', 'Enter worker name here').attr('required\
', 'required').css({float: 'right'}));
    }


    /* Given an hitId, add to the form. */
    hitId = queryParameters.hitId;

    if (hitId) {
      $("#mturk").append($('<input/>').attr('type', 'hidden').attr('name', 'hitId').val(hitId));
    } else {
	hitId = 'none';
    }

}
