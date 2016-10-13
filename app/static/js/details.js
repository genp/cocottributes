/**
 * Return JavaScript object of query parameters
*/
function getUrlVars()
{
    var vars = [], hash;
    var hashes = window.location.href.
                 slice(window.location.href.indexOf('?') + 1).split('&');

    for(var i = 0; i < hashes.length; i++) {
        hash = hashes[i].split('=');
        vars[hash[0]] = hash[1];
    }

    return vars;
}


function get_location()
{
    $.ajax( {
        url: '//freegeoip.net/json/',
        type: 'POST',
        dataType: 'jsonp',
        success: function(location) {
            // city = location.city;
            nationality = location.country_name;
            // longitude = location.longitude;
            // longitude = location.latitude;
            ip = location.ip;
        }
    } );
}
