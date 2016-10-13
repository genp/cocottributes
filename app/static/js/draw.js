function renderSegmentation(pts, ctx){
// ===========================
// pts:  list of points in this mask
// ctx:    canvas (HTML DOM) context (link)
// ===========================

     // set color for each object
     var r = clr[cur_clr][0]
     var g = clr[cur_clr][1]
     var b = clr[cur_clr][2]
     ctx.fillStyle = 'rgba('+r+','+g+','+b+',0.7)';
     update_cur_clr();
     ctx.beginPath();
     ctx.moveTo(parseFloat(pts[0]), parseFloat(pts[1]));
     for (j=0; j<pts.length; j += 2){
         px = pts[j];
         py = pts[j+1];
         // let's draw!!!!
         ctx.lineTo(parseFloat(px), parseFloat(py));
     }

     ctx.lineWidth = 3;
     ctx.closePath();
     ctx.fill();
     ctx.strokeStyle = 'black';
     ctx.stroke();

};

function renderOutline(pts, ctx, clr){
// ===========================
// pts:  list of points in this mask
// ctx:    canvas (HTML DOM) context (link)
// ===========================
     clr = typeof clr !== 'undefined' ? clr : 'white';
     ctx.beginPath();
     ctx.moveTo(parseFloat(pts[0]), parseFloat(pts[1]));
     for (j=0; j<pts.length; j += 2){
         px = pts[j];
         py = pts[j+1];
         // let's draw!!!!
         ctx.lineTo(parseFloat(px), parseFloat(py));
     }

     ctx.lineWidth = 5;
     ctx.closePath();
     ctx.strokeStyle = 'black';
     ctx.stroke();

     ctx.beginPath();
     ctx.moveTo(parseFloat(pts[0]), parseFloat(pts[1]));
     for (j=0; j<pts.length; j += 2){
         px = pts[j];
         py = pts[j+1];
         // let's draw!!!!
         ctx.lineTo(parseFloat(px), parseFloat(py));
     }

     ctx.lineWidth = 2;
     ctx.closePath();
     ctx.strokeStyle = clr;
     ctx.stroke();

};


function canvas_annotation(cname, patches, img_id, outlineOnly, width, crop){
    console.log('annotating canvas...')

    var canvas = $('#'+cname);
    var context = canvas[0].getContext('2d');
    var imageObj = new Image();

    imageObj.src = '/image/'+img_id.toString();

    jQuery.data(canvas, 'imageObj', imageObj);

    imageObj.onload = function() {

        var h = this.height;
        var w = this.width;

        jQuery.data(canvas, 'patches', patches);
        jQuery.data(canvas, 'outlineOnly', outlineOnly);
        jQuery.data(canvas, 'crop', crop);

	var sx = 0; 
	var sy = 0; 
	var sw = 0;
	var sh = 0;
        var scale = 1.0;

	if (crop == 1) {

	    // the patch's [x,y,h,w] and img's [w,h] will be in patches[p][1-6], patches[p][0] is the segmentation
	    x = patches[0][1];
	    y = patches[0][2];
	    pw = patches[0][3];
	    ph = patches[0][4];
	    sx = Math.max(0,x-pw*0.25);
	    sy = Math.max(0,y-ph*0.25);
	    sw = Math.min(2*pw, w-sx);
	    sh = Math.min(2*ph, h-sy);

	    w = sw/2;
	    h = sh/2;
	    var sc = w/sw;
	    for(p=0; p<patches.length; p++){
    		var pts = patches[p][0];
    		for (ind = 0; ind<pts.length; ind=ind+2){
    		    pts[ind] = (pts[ind]-sx)*sc;
    		    pts[ind+1] = (pts[ind+1]-sy)*sc;
		}
            }

	} else {
            h = this.height;
            w = this.width;
	}


        if(h >= w){

            scale = width/h;
            w = Math.floor(w * scale);
            h = width;
        } else {

            scale = width/w;
            h = Math.floor(h * scale);
            w = width;
        }


        jQuery.data(canvas, 'w', w);
        jQuery.data(canvas, 'h', h);
  	canvas.attr({'width':w, 'height':h});
        jQuery.data(canvas, 'sx', sx);
        jQuery.data(canvas, 'sy', sy);
        jQuery.data(canvas, 'sw', sw);
        jQuery.data(canvas, 'sh', sh);

	draw_annotation(context, imageObj, w, h, patches, scale, outlineOnly, 'white', crop, sx, sy, sw, sh);

	var is_green = false;
	var is_hover = false;
	setInterval(function () {
	    if (!is_hover) {
		if (is_green) {
		    draw_annotation(context, imageObj, w, h, patches, 1.0, outlineOnly, 'white', crop, sx, sy, sw, sh);
		    is_green = false;
		} else {
		    draw_annotation(context, imageObj, w, h, patches, 1.0, outlineOnly, '#bbb', crop, sx, sy, sw, sh);
		    is_green = true;
		}
	    }
	}, 500); 
	
        canvas.hover(function(){
            // draw plain image
	    var context = canvas[0].getContext('2d');
	    if (jQuery.data(canvas, 'crop') == 1) {
		context.drawImage(jQuery.data(canvas, 'imageObj'), 
				  jQuery.data(canvas, 'sx'), jQuery.data(canvas, 'sy'),
				  jQuery.data(canvas, 'sw'), jQuery.data(canvas, 'sh'),
				  0, 0, 
				  jQuery.data(canvas, 'w'), jQuery.data(canvas, 'h'));
	    } else {
		context.drawImage(jQuery.data(canvas, 'imageObj'), 0, 0, 
				  jQuery.data(canvas, 'w'), jQuery.data(canvas, 'h'));
	    }
	    is_hover = true;
        }, function(){
	    var context = canvas[0].getContext('2d');
	    // draw image with annotation
	    draw_annotation(context, jQuery.data(canvas, 'imageObj'), 
			    jQuery.data(canvas, 'w'), jQuery.data(canvas, 'h'),
			    jQuery.data(canvas, 'patches'), 1.0, jQuery.data(canvas, 'outlineOnly'), 'white',
			    jQuery.data(canvas, 'crop'), jQuery.data(canvas, 'sx'),
			    jQuery.data(canvas, 'sy'), jQuery.data(canvas, 'sw'), jQuery.data(canvas, 'sh'));
	    is_hover = false;
        });
    }; 
};

function draw_annotation(context, imageObj, w, h, patches, scale, outlineOnly, clr, crop, sx, sy, sw, sh){
    if(crop == 1){
	context.drawImage(imageObj, sx, sy, sw, sh, 0, 0, w, h);

    } else {
        context.drawImage(imageObj, 0, 0, w, h);
    }

    for (p=0; p<patches.length; p++){
      var pts = patches[p][0];
	  for (ind = 0; ind<pts.length; ind++){
          pts[ind] = pts[ind]*scale;
	  }
      if(outlineOnly == 1){
	  if (clr != -1) { 
  	      renderOutline(pts, context, clr);
	      }
      } else if (outlineOnly == 2) {
  	      renderSegmentation(pts, context);
      } 
    }

 
};

function make_grid(dname, patches, images, outlineOnly, clickable, width, crop){
    var boxclassname = clickable ? 'clickablebox' : 'box';
    var outerboxclassname = clickable ? 'clickableOuterBox' : 'outerBox';
    var canvasclassname = crop ? 'cropped' : 'notcropped';
    for(idx=0; idx< patches.length; idx++){
	var newDivOuter = $('<div/>',{
	                class: outerboxclassname,
	                id: 'outerBox'+idx.toString()
                    });
	var newDiv = $('<div/>',{
                        class: boxclassname
                    });
	var cname = 'c'+idx.toString()
        var newCanvas = $('<canvas/>',{
                        id: cname,
	                class: canvasclassname
                    });

	newDivOuter.append(newDiv);
	newDiv.append(newCanvas);
        $('#'+dname).append(newDivOuter);

	canvas_annotation(cname, [patches[idx]], images[idx], outlineOnly, width, crop)
    }    
        if (clickable) {

            $('canvas').on('click', function(e){
              e.preventDefault();
              $(this).toggleClass('positive');
	      var bidx = $(this).closest('canvas').prop("id").split('c')[1];
	      if (!$('#'+bidx+'_0').prop("checked") & !$('#'+bidx+'_1').prop("checked")) {
	      	  $('#'+bidx+'_0').prop("checked", true);
	      } else {
		  $('#'+bidx+'_0').prop("checked", !$('#'+bidx+'_0').prop("checked"));
		  $('#'+bidx+'_1').prop("checked", !$('#'+bidx+'_1').prop("checked"));
	      }
              update_selections();


            });

        }

};

function make_checkbox_grid(options, num_boxes){

    for(idx=0; idx< num_boxes; idx++){
	var cname = 'attrs'+idx.toString()
	var newFieldset = $('<fieldset/>',{
	                id: cname,
                        class: 'checks'
                    });

	var newDiv = $('<div/>',{
                        class: 'box'
                    });

        $('#outerBox'+idx.toString()).append(newDiv);
	newDiv.append('<h3> Image '+(idx+1).toString()+'</h3>');
	newDiv.append(newFieldset);

	var checkList = $('<ul/>', {class: "checkbox" });
	newFieldset.append(checkList);

	for(jdx=0; jdx < options.length; jdx++){
	    var id = idx.toString()+'_'+jdx.toString();
	    if ( options[jdx] == 'None' ) {
		id = 'none';
		}
	    var newListItem = $('<li/>');
	    var newCB = $( '<input>', {
		            type: 'checkbox',
		            class: 'opt-check',
		            id: id
		        });
	    var newLabel = $( '<label/>', {
		            for: id,
		            class: 'side-label',
		            html: options[jdx],
		            title: ''
		        });
	    newLabel.tooltip({ content: options[jdx] + ': ' + String(attr_defns[jdx]) });//+ '<img src="https://cs.brown.edu/~gen/Attributes/defn_images_small/climbing1.jpg" />' });
	    newListItem.append(newCB);
	    newListItem.append(newLabel);
	    checkList.append(newListItem);

	}
	var id = 'none'+idx.toString();

        var newListItem = $('<li/>');
        var newCB = $( '<input>', {
    	            type: 'checkbox',
    	            class: 'none-check',
    	            id: id
    	        });
        var newLabel = $( '<label/>', {
    	            for: id,
	            id: id,
    	            class: 'none-label',
    	            html: 'None'
    	        });
        newListItem.append(newCB);
        newListItem.append(newLabel);
        checkList.append(newListItem);


    }


};


function add_checkboxes(options, num_boxes){

    for(idx=0; idx< num_boxes; idx++){
	var cname = 'attrs'+idx.toString()
	var newFieldset = $('<fieldset/>',{
	                id: cname,
                        class: 'checks'
                    });

	var newDiv = $('<div/>',{
                        class: 'checkboxbox'
                    });

        $('#outerBox'+idx.toString()+' > div').append(newDiv);
	newDiv.append('<h3> Image '+(idx+1).toString()+'</h3>');
	newDiv.append(newFieldset);

	var checkList = $('<ul/>', {class: "checkbox" });
	newFieldset.append(checkList);

	for(jdx=0; jdx < 2; jdx++){
	    var id = idx.toString()+'_'+jdx.toString();
	    var checkclass = jdx == 0 ? 'opt-check' : 'none-check';
	    if ( options[jdx] == 'None' ) {
		id = 'none';
		}
	    var newListItem = $('<li/>');
	    var newCB = $( '<input>', {
		            type: 'checkbox',
		            class: checkclass,
		            id: id
		        });
	    var newLabel = $( '<label/>', {
		            for: id,
		            class: 'side-label',
		            html: options[jdx],
		            title: ''
		        });

	    newListItem.append(newCB);
	    newListItem.append(newLabel);
	    checkList.append(newListItem);

	}
    }
};

function update_selections(){

    num_clicks = $('.positive').length;
    $('#clicks').html(num_clicks);
};


