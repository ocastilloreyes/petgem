.. _Contact:

Contact
=======
| Octavio Castillo-Reyes
| Tel: +34 934137992
| email: `octavio.castillo@bsc.es`_
| Location: Nexus II building - third floor C/ Jordi Girona, 29. Barcelona 08034

.. _octavio.castillo@bsc.es: octavio.castillo@bsc.es

View map
========

.. raw:: html

    <style>
      #map {
      height: 400px;
      width: 600px;
      }
    </style>

    <div id="map"></div>
    <script>
      function initMap() {
        var uluru = {lat: 41.388062, lng: 2.114986};
        var map = new google.maps.Map(document.getElementById('map'), {
          zoom: 15,
          center: uluru
        });
        var marker = new google.maps.Marker({
          position: uluru,
          map: map
        });
      }
    </script>
    <script async defer
    src="https://maps.googleapis.com/maps/api/js?key=AIzaSyDD5WVjZgm82OKuirq0oB2TvNE4REKNngU&callback=initMap">
    </script>
