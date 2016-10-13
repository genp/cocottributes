function load_patches(patches) {
    console.log('load patches');
    for(var i =0; i < Object.keys(patches).length; i++){
        imgName = Object.keys(patches)[i];
        for(var j=0; j < patches[imgName].length; j++){
                    array = patches[imgName][j];

                    x = array[0];
                    y = array[1];
                    size = array[2];
            	    c = array[3];

                    image = new Image();
                    image.tempname = imgName;
                    image.tempx = x;
                    image.tempy = y;
                    image.tempsize = size;
	            image.classifier = c;
                    image.onload = function(){

			$("#patches").append($('<div>', { id: this.classifier, class: "bbox", style: "height: 100px; width: 100px; background-image : url(/blob/" +this.tempname+"); background-size : " + (100.0 * this.width/this.tempsize) + "px; background-position : -" + (100.0 * this.tempy / this.tempsize) + "px -" + (100.0 * this.tempx / this.tempsize) + "px;" }));
			    $("#"+this.classifier).click( function () {
				$(location).attr('href',"/classifier/"+this.id);
			    });
                    }

                    image.src = "/blob/" + imgName;

        }
    }

}



// images is a list of blob ids, and div_name is the place to show them
function load_images(images, div_name) {

    for(var i =0; i < images.length; i++){
        imgName = images[i];
	$("#"+div_name).append($('<div>', { id: imgName+"_div"}));
	$("#"+imgName+"_div").html($('<img>', { id: imgName, src : "/blob/" + imgName, width: "300"}));

    }
    
}
