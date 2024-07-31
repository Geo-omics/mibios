"""
Extra attributes for fields of the Sample model

This module is compiled via the compile_extra_field_attributes command.  The
module file should be tracked by revision control but not be manually edited.
"""

extra_sample_field_attrs = {
    'temp': {'verbose_name': 'Temperature', 'unit': '°C'},
    'depth': {'verbose_name': 'Sample collection depth', 'unit': 'm'},
    'depth_sediment': {'verbose_name': 'Sediment depth', 'unit': 'm'},
    'depth_location': {'verbose_name': 'Depth at sampling location', 'unit': 'm'},
    'env_local_scale': {},
    'env_medium': {'verbose_name': 'Sample material', 'pseudo_unit': 'ENVO ontology'},
    'geo_loc_name': {'verbose_name': 'Location', 'pseudo_unit': 'GAZ ontology'},
    'gaz_id': {},
    'latitude': {'verbose_name': 'Latitude', 'pseudo_unit': 'decimal degrees'},
    'longitude': {'verbose_name': 'Longitude', 'pseudo_unit': 'decimal degrees'},
    'collection_timestamp': {'verbose_name': 'collection timestamp'},
    'noaa_site': {'verbose_name': 'NOAA site'},
    'env_broad_scale': {'verbose_name': 'ENVO Ontology ID', 'pseudo_unit': 'ENVO ontology'},
    'size_frac_low': {'verbose_name': 'Pre-filter pore size', 'unit': 'μm'},
    'size_frac_up': {'verbose_name': 'Filter retained pore size', 'unit': 'μm'},
    'conduc': {'verbose_name': 'Conductivity', 'unit': 'μS/cm'},
    'calcium': {'verbose_name': 'Calcium', 'unit': 'mg/L'},
    'diss_oxygen': {'verbose_name': 'Dissolved oxygen', 'unit': 'mg/L'},
    'potassium': {'verbose_name': 'Potassium', 'unit': 'mg/L'},
    'phosphate': {'verbose_name': 'Phosphate', 'unit': 'mg/L'},
    'magnesium': {'verbose_name': 'Magnesium', 'unit': 'mg/L'},
    'ammonium': {'verbose_name': 'Ammonium', 'unit': 'mg/L'},
    'secchi': {'verbose_name': 'Secchi depth', 'unit': 'm'},
    'part_microcyst': {'verbose_name': 'Particulate microcystin', 'unit': 'μg/L'},
    'diss_microcyst': {'verbose_name': 'Dissolved Microcystin', 'unit': 'μg/L'},
    'tot_microcyst_lcmsms': {'verbose_name': 'Total microcystin determined by LCMSMS', 'unit': 'ug/L'},
    'ext_phyco': {'verbose_name': 'Phycocyanin', 'unit': 'μg/L'},
    'ext_microcyst': {'verbose_name': 'Microcystin', 'unit': 'μg/L'},
    'ext_anatox': {'verbose_name': 'Anatoxin', 'unit': 'μg/L'},
    'chlorophyl': {'verbose_name': 'Chlorophyll', 'unit': 'μg/L'},
    'total_phos': {'verbose_name': 'Total phosphorus', 'unit': 'μg/L'},
    'diss_phos': {'verbose_name': 'Total dissolved phosphorus', 'unit': 'μg/L'},
    'soluble_react_phos': {'verbose_name': 'Soluble reactive phosphorus', 'unit': 'μg/L'},
    'ammonia': {'verbose_name': 'Ammonia', 'unit': 'μg/L'},
    'nitrate_nitrite': {'verbose_name': 'Nitrate and nitrite', 'unit': 'mg/L'},
    'nitrate': {'verbose_name': 'Nitrate', 'unit': 'mg/L'},
    'nitrite': {'verbose_name': 'Nitrite', 'unit': 'μg/L'},
    'urea': {'verbose_name': 'Urea', 'unit': 'μg/L'},
    'part_org_carb': {'verbose_name': 'Particulate organic carbon', 'unit': 'mg/L'},
    'part_org_nitro': {'verbose_name': 'Particulate organic nitrogen', 'unit': 'mg/L'},
    'diss_org_carb': {'verbose_name': 'Dissolved organic carbon', 'unit': 'mg/L'},
    'col_dom': {'verbose_name': 'DOM absorbance at 400 nm', 'unit': 'm-1'},
    'h2o2': {'verbose_name': 'hydrogen peroxide', 'unit': 'nm'},
    'suspend_part_matter': {'verbose_name': 'Suspended particulate matter', 'unit': 'mg/L'},
    'suspend_vol_solid': {'verbose_name': 'Suspended volitile solids', 'unit': 'mg/L'},
    'turbidity': {'verbose_name': 'Turbidity', 'unit': 'NTU'},
    'attenuation': {'verbose_name': 'Attenuation', 'unit': 'm-1'},
    'transmission': {'verbose_name': 'Transmission', 'unit': '%'},
    'microcystis_count': {'verbose_name': 'Microcystis cell count', 'unit': 'cells/mL'},
    'planktothrix_count': {'verbose_name': 'Planktothrix cell count', 'unit': 'cells/mL'},
    'anabaena_d_count': {'verbose_name': 'Anabaena cell count', 'unit': 'cells/mL'},
    'cylindrospermopsis_count': {'verbose_name': 'Cylidrospermopsis cell count', 'unit': 'cells/mL'},
    'ice_cover': {'verbose_name': 'Ice cover', 'unit': '%'},
    'chlorophyl_fluoresence': {'verbose_name': 'Chlorophyll a fluorescence', 'unit': 'rel AU'},
    'silicate': {'verbose_name': 'Silicate', 'unit': 'umol/L'},
    'is_neg_control': {'verbose_name': 'Negative control', 'pseudo_unit': 'TRUE / FALSE'},
    'is_pos_control': {'verbose_name': 'Mock community', 'pseudo_unit': 'TRUE / FALSE'},
    'samp_vol_we_dna_ext': {'verbose_name': 'Volume filtered', 'unit': 'mL'},
    'filt_duration': {'verbose_name': 'Filtration duration', 'pseudo_unit': 'hh:mm:ss'},
    'par': {'verbose_name': 'Photosynthetically active radiation', 'unit': 'μE/m2/s'},
    'qPCR_total': {'verbose_name': 'qPCR; Phytoxigene total cyanobacteria', 'unit': 'copies/mL'},
    'qPCR_mcyE': {'verbose_name': 'qPCR; Phytoxigene mcyE', 'unit': 'copies/mL'},
    'qPCR_sxtA': {'verbose_name': 'qPCR; Phytoxigene sxtA', 'unit': 'copies/mL'},
    'gold_analysis_id': {'verbose_name': 'GOLD analysis projectID'},
    'gold_seq_id': {'verbose_name': 'GOLD sequencing projectID'},
    'wind_speed': {},
    'wave_height': {},
    'sky': {},
    'tot_nit': {'verbose_name': 'Total nitrogen', 'unit': 'umol/L'},
    'green_algae': {'verbose_name': 'Green algae', 'unit': 'ug/L'},
    'bluegreen': {'verbose_name': 'Bluegreen algae', 'unit': 'ug/L'},
    'diatoms': {'verbose_name': 'Diatoms', 'unit': 'ug/L'},
    'crypto': {'verbose_name': 'Cryptophyta', 'unit': 'ug/L'},
    'sortchem': {},
    'phyco_fluoresence': {'verbose_name': 'Phycocyanin fluoresence ', 'unit': 'RFU'},
    'env_canada_site': {'verbose_name': 'Environment Canada Sampling site'},
    'env_kenya_site': {'verbose_name': 'Environment Kenya Sampling site'},
    'amended_site_name': {'verbose_name': 'Amended sampling location name if not the same as the original site name'},
    'salinity': {'verbose_name': 'Salinity measured using a sonde', 'unit': 'parts per thousand (PPT)'},
    'atmospheric_temp': {'verbose_name': 'atmospheric temperature', 'unit': 'C'},
    'particulate_cyl': {'verbose_name': 'particulate cylindrospermopsin, measured via ELISA', 'unit': 'ng/mL'},
    'station_description': {'verbose_name': 'station description about location'},
    'replicate_id': {'verbose_name': 'replicate id if metagenomic samples sequenced in replicate'},
    'orp': {'verbose_name': 'Redox potential-ORP', 'unit': 'millivolts (mV)'},
    'cyano_sonde': {'verbose_name': 'AlgaeTorch Cyanobacterial abundance detection (cyanobacteria chlorophyll-a)', 'unit': 'c/L'},
    'total_sonde': {'verbose_name': 'AlgaeTorch total chlorophyll-a detection', 'unit': 'c/L'},
    'sampling_device': {'verbose_name': 'Method or device employed for collecting sample'},
}
