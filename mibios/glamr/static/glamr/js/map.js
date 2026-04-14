async function get_map_points() {
    let points;
    let url = new URL(window.location);
    url.searchParams.delete('mapmode');
    url.searchParams.append('mapmode', 'only');
    let resp;
    try {
        resp = await fetch(url, {method: "GET"});
    } catch (e) {
        msg = 'failed fetching GLAMR map points';
        console.error(msg + ": " + e);
        return {'error': msg};
    }

    if (resp.ok) {
        const resp2 = resp.clone()
        try {
            return await resp.json();
        } catch (e) {
            msg = "failed parsing GLAMR map point data";
            console.error(msg + ": " + e);
            resp_txt = await resp2.text();
            console.debug("data starts thus:\n" + resp_txt.slice(0, 100));
            return {'error': msg};
        }
    } else {
        msg = "failed fetching map points (status: " + resp.status + ')';
        try {
            resp_txt = await resp.text();
            console.debug("begin of content:\n" + resp_txt.slice(0, 100));
        } catch (e) {
            console.debug("failed getting content: " + e);
        }
        return {'error': msg};
    }
};


function show_error(map, msg, extra='') {
    /* Propagate some errors message to a little text widget on the map */
    console.error(msg + ' ' + extra);
    var c = L.control({position: 'bottomright'});
    c.onAdd = function () {
        let div = L.DomUtil.create('div', 'info legend p-1');
        div.innerText = "Error: " + msg;
        return div;
    };
    c.addTo(map);
}


async function populate_map(L, fit_map_to_points, defer, hide_if_empty) {
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

    var map_points;
    if (defer) {
        const map_data = await get_map_points();
        if ('error' in map_data) {
            show_error(map, map_data.error);
            return;
        } else {
            map_points = map_data.points;
        }
    } else {
        try {
            map_points = JSON.parse(document.getElementById('map_data').textContent);
        } catch (e) {
            show_error(map, 'failed parsing map point data', e);
            return
        }
    }

    if (map_points == undefined) {
        show_error(map, 'map_points are missing');
        return;
    }

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

                    var sample_type = map_points[i].sample_type
                    if (sample_type) {
                        sample_type = sample_type[0].toUpperCase() + sample_type.slice(1);
                    } else {
                        sample_type = '(missing)';
                    }
                    sample_type = "Sample Type: " + sample_type + "<br/>";

                    var collection_timestamp = new Date(map_points[i].collection_timestamp);
                    var sample_date = "Collection Date: " + collection_timestamp.toLocaleDateString(0,{year:'numeric', month:'2-digit', day: '2-digit'});
                    var others = "";
                    if (map_points[i].others) {others=map_points[i].others;}

                    var iconType = "other-icon";
                    if (map_points[i].types_at_location) {
                        iconType = map_points[i].types_at_location + "-icon";
                    }
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
}

populate_map(
    L,
    true ? document.currentScript.dataset.fit_maps_to_points == 'True' : false,
    true ? document.currentScript.dataset.deferred == 'True' : false,
    true ? document.currentScript.dataset.hide_empty_map == 'True' : false,
);
