
var hex_clr = colorbrewer['Set3'][12];
var clr = load_clrs(hex_clr);
var cur_clr = 0;

function load_clrs(hexclrs){
    var res = []
    for (i=0; i<hexclrs.length; i++){
	val = cutHex(hexclrs[i]);
	r = hexToR(val);
	g = hexToG(val);
	b = hexToB(val);
	res.push([r,g,b]);
    }
    return res
};

function hexToR(h) {return parseInt((h).substring(0,2),16)};
function hexToG(h) {return parseInt((h).substring(2,4),16)};
function hexToB(h) {return parseInt((h).substring(4,6),16)};
function cutHex(h) {return (h.charAt(0)=="#") ? h.substring(1,7):h};



function update_cur_clr(){
    cur_clr = (cur_clr + 1) % clr.length;

};


