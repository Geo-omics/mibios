// adapted from https://www.paulox.net/2020/12/08/maps-with-django-part-1-geodjango-spatialite-and-leaflet/#adding-an-empty-web-map
const copy = "&copy; <a href='https://www.openstreetmap.org/copyright'>OpenStreetMap</a> contributors";
const url = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const layer = L.tileLayer(url, { attribution: copy });
const map = L.map("map", { 
	layers: [layer], 
	center: [45, -85],
	zoom: 5.3,
	scrollWheelZoom: false 
});
L.control.scale().addTo(map);
const map_points = JSON.parse(
  document.currentScript.nextElementSibling.textContent
);
var markers = [];

// adapted from https://gis.stackexchange.com/questions/195422/create-map-using-leaflet-and-json
for (var i in map_points) {
	if(map_points[i].latitude & map_points[i].longitude){
		var lat_long = L.latLng({ lat: map_points[i].latitude, lng: map_points[i].longitude });
   		var sample_url = "Sample: <a href='" + map_points[i].sample_url + "'>" + map_points[i].sample_name + "</a>"
    	var dataset_url = "Dataset: <a href='" + map_points[i].dataset_url + "'>" + map_points[i].dataset_name + "</a>"
    	
    	var marker = L.marker(lat_long)
    	marker.bindPopup(sample_url + "<br/>" + dataset_url).openPopup();
    	//marker.addTo(map);
    	markers.push(marker);
	}
}

var group = L.featureGroup(markers).addTo(map);
map.fitBounds(group.getBounds());
