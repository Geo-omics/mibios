// adapted from https://www.paulox.net/2020/12/08/maps-with-django-part-1-geodjango-spatialite-and-leaflet/#adding-an-empty-web-map
const copy = "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors";
const url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const layer = L.tileLayer(url, { attribution: copy });
const map = L.map("map", { 
	layers: [layer], 
	center: [45.10, -84.29],
	zoom: 6,
	scrollWheelZoom: false 
});
L.control.scale().addTo(map);
const map_points = JSON.parse(
  document.currentScript.nextElementSibling.textContent
);

// leaflet legend help from https://gis.stackexchange.com/questions/133630/adding-leaflet-legend
var legend = L.control({position: 'bottomright'});
legend.onAdd = function (map) {
	var div = L.DomUtil.create('div', 'info legend p-1');
	
	categories = ['Amplicon','Metagenome','Metatranscriptome'];
	div.innerHTML = '<strong>Sample Type</strong><br/><i class="bi bi-circle-fill m-1 amplicon-legend"></i>Amplicon<br/><i class="bi bi-circle-fill m-1 metagenome-legend"></i>Metagenome<br/><i class="bi bi-circle-fill m-1 metatranscriptome-legend"></i>Metatranscriptome<br/>';
	
	return div;
};
legend.addTo(map);

var markers = [];

// adapted from https://gis.stackexchange.com/questions/195422/create-map-using-leaflet-and-json
for (var i in map_points) {
	if(map_points[i].latitude & map_points[i].longitude){
                var lat_long = L.latLng({ lat: map_points[i].latitude, lng: map_points[i].longitude });

                var sample_name = map_points[i].sample_name ? map_points[i].sample_name  : (map_points[i].sample_id ? map_points[i].sample_id : (map_points[i].biosample ? map_points[i].biosample : "<i>Name not provided</i>"))
                var sample_url = "Sample: <a href='" + map_points[i].sample_url + "'>" + sample_name + "</a>"

                var dataset_url = "Dataset: <a href='" + map_points[i].dataset_url + "'>" + map_points[i].dataset_name + "</a>"

                var sample_type_string = map_points[i].sample_type[0].toUpperCase() + map_points[i].sample_type.slice(1)
                var sample_type = "Sample Type: " + sample_type_string + "<br/>"

                var collection_timestamp = new Date(map_points[i].collection_timestamp)
                var sample_date = "Collection Date: " + collection_timestamp.toLocaleDateString(0,{year:'numeric', month:'2-digit', day: '2-digit'})

                var iconType = "amplicon-icon";
                if(map_points[i].sample_type == "metagenome"){
                        iconType = "metagenome-icon";
                }
                else if(map_points[i].sample_type == "metatranscriptome"){
                        iconType="metatranscriptome-icon"
                }

                var iconMarker = L.divIcon({className: iconType});

                var marker = L.marker(lat_long,{icon: iconMarker})
                marker.bindPopup(sample_url + "<br/>" + sample_date + "<br/>" + sample_type + dataset_url).openPopup();
                markers.push(marker);
	}
}

var group = L.featureGroup(markers).addTo(map);
