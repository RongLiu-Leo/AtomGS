window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE_OURS = "./static/interpolation/ours/shrunken";
var INTERP_BASE_GS = "./static/interpolation/gs/shrunken";
var INTERP_FULL_OURS = "./static/interpolation/ours/full";
var INTERP_FULL_GS = "./static/interpolation/gs/full";

var NUM_INTERP_FRAMES = 200;

var interp_images_ours = [];
var interp_images_gs = [];
var interp_images_full_ours = [];
var interp_images_full_gs = [];

function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path_ours = INTERP_BASE_OURS + '/' + String((i+1)).padStart(4, '0') + '0.png';
    var path_gs = INTERP_BASE_GS + '/' + String((i+1)).padStart(4, '0') + '0.png';

    var path_full_ours = INTERP_FULL_OURS + '/' + String((i+1)).padStart(4, '0') + '0.png';
    var path_full_gs = INTERP_FULL_GS + '/' + String((i+1)).padStart(4, '0') + '0.png';

    interp_images_ours[i] = new Image();
    interp_images_ours[i].src = path_ours;

    interp_images_gs[i] = new Image();
    interp_images_gs[i].src = path_gs;

    interp_images_full_ours[i] = new Image();
    interp_images_full_ours[i].src = path_full_ours;

    interp_images_full_gs[i] = new Image();
    interp_images_full_gs[i].src = path_full_gs;
  }
}

function setInterpolationImage(i) {
  var ours = document.getElementById('image-ours');
  var gs = document.getElementById('image-gs');
  let parts = ours.src.split('/');
  if (parts[parts.length-2] == "shrunken"){
      ours.src = interp_images_ours[i].src;
      gs.src = interp_images_gs[i].src;
  }else{
      ours.src = interp_images_full_ours[i].src;
      gs.src = interp_images_full_gs[i].src;
  }

  
}

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})
