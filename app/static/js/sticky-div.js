function sticky_relocate() {
    var window_top = $(window).scrollTop();
    var div_top = $('#img_grid').offset().top;
    if (window_top > div_top)
        $('#example-container').addClass('sticky')
      else
          $('#example-container').removeClass('sticky');
}
$(function() {
    $(window).scroll(sticky_relocate);
    sticky_relocate();
});
