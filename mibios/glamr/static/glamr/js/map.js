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

// load data and variables passed to the script
const map_points = JSON.parse(
  document.currentScript.nextElementSibling.textContent
);
const data = document.currentScript.dataset;
const fit_map_to_points = data.fit_map_to_points;

// leaflet legend help from https://gis.stackexchange.com/questions/133630/adding-leaflet-legend
var legend = L.control({position: 'bottomright'});
legend.onAdd = function (map) {
    let div = L.DomUtil.create('div', 'info legend p-1');
	
    const categories = ['amplicon','metagenome','metatranscriptome'];
    const items = ["<strong>Sample Type</strong>"];

    for (let i=0; i<categories.length; i++) {
        items.push(
            '<i class="' + categories[i] + '-icon" '
            + 'style="display:inline-block;height:13px;width:13px;vertical-align:middle;margin-left:0.2rem;margin-right:0.2rem;">'
            + '</i>'
            + categories[i].slice(0, 1).toUpperCase() + categories[i].slice(1)
        );
    }
    div.innerHTML = items.join('<br>');
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
                var others = "";
                if (map_points[i].others) {others=map_points[i].others;}

                var iconType = map_points[i].types_at_location + "-icon";
                var iconMarker = L.divIcon({className: iconType});

                var marker = L.marker(lat_long,{icon: iconMarker})
                marker.bindPopup(sample_url + "<br/>" + sample_date + "<br/>" + sample_type + dataset_url + others).openPopup();
                markers.push(marker);
	}
}

var group = L.featureGroup(markers).addTo(map);

// set map zoom/bounds using the points we added instead of the default
// showing the Great Lakes, intended for the frontpage
if(fit_map_to_points){
    var maxzoom = null;
    if (markers.length <= 1) {
        // for single points zoom out a bit, else the fitBounds zooms in
        // all the way, usually showing just water
        maxzoom = 8;
    }
    map.fitBounds(group.getBounds(), {maxZoom: maxzoom});
}
