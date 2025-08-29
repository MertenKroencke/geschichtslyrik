import pandas as pd
import re
from scipy import stats

import plotly.express as px
import plotly.graph_objects as go
from plotly.validators.scatter.marker import SymbolValidator

# Angabe, welche Spalte in Annotationstabelle welche Kategorie repräsentiert
pos_id = 0
pos_page = 1
pos_authors_names = 3
pos_gnds = 4
pos_ergaenzung = 5
pos_lifetimes_birth = 6
pos_lifetimes_death = 7
pos_title_single = 8
pos_title_unified = 9
pos_text_written = 10
pos_text_published = 11
pos_analysis_start = 14
pos_geschichtslyrik = 14
pos_empirisch = 15
pos_theoretisch = 16
pos_gattung = 17
pos_sprechinstanz_markiert = 18
pos_sprechinstanz_zeitebene = 19
pos_sprechakt = 20
pos_tempus = 21
pos_zeitdominanz = 22
pos_zeitebenen = 23
pos_zeit_fixierbarkeit = 24
pos_time_start = 25
pos_time_end = 26
pos_anachronismus = 27
pos_gegenwartsbezug = 28
pos_grossraum = 29
pos_mittelraum = 30
pos_kleinraum = 31
pos_inhalt_typ = 32
pos_themes = 33
pos_subject = 34
pos_entity = 35
pos_entity_bewertung = 36
pos_themes_bewertung = 37
pos_patriotismus = 38
pos_heroismus = 39
pos_feindbilder = 40
pos_religion = 41
pos_marker_pers = 42
pos_marker_time = 43
pos_marker_place = 44
pos_marker_object = 45
pos_ueberlieferung = 46
pos_ueberlieferung_bewertung = 47
pos_geschichtsvorstellung = 48
pos_geschichtsvorstellung_bewertung = 49
pos_rel_history = 50
pos_sicherheit = 51
pos_reim = 52
pos_metrum = 53
pos_verfremdung = 54
pos_anschaulichkeit = 55
pos_hist_darstellungsweise = 56
pos_analysis_end = 56

def hex_to_rgba(h, alpha = 0.2):
    rgba = tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])
    rgba_string = f"rgba{rgba}"
    return rgba_string
plotly_colors_rgba = [hex_to_rgba(x) for x in px.colors.qualitative.Plotly]

def get_rating_table(meta, mode):
    meta = meta.reset_index()
    author_list, title_list, full_list, type_list, rating_list = [], [], [], [], []
    
    if mode == 'entity':
        rated_data_full = 'entity_full'
        rated_data_simple = 'entity_simple'
        rating_data = 'entity_bewertung'
    if mode == 'themes':
        rated_data_full = 'stoffgebiet'
        rated_data_simple = 'stoffgebiet'
        rating_data = 'stoffgebiet_bewertung'
    if mode == 'ueberlieferung':
        rated_data_full = 'ueberlieferung'
        rated_data_simple = 'ueberlieferung'
        rating_data = 'ueberlieferung_bewertung'
    if mode == 'geschichtsauffassung':
        rated_data_full = 'geschichtsauffassung'
        rated_data_simple = 'geschichtsauffassung'
        rating_data = 'geschichtsauffassung_bewertung'
        
    for element in meta.iloc:
        if element[rated_data_full] and str(element[rated_data_full]) != 'nan': # gibt bewertete phänomene                  
            if mode == 'entity': 
                this_rated_full_split = [re.sub("\\]", "", re.sub("[0-9] \\[", "", x)) for x in element[rated_data_full].split(" + ")]
                this_rated_simple_split = element[rated_data_simple].split(" + ")
                this_rating_data_split = [x.strip() for x in element[rating_data].split(" + ")]
            if mode == 'themes':
                this_rated_full_split = element[rated_data_full].split(" + ")
                this_rated_simple_split = element[rated_data_simple].split(" + ")
                this_rating_data_split = [x.strip() for x in element[rating_data].split(" + ")]
            if mode == 'ueberlieferung':
                this_rated_full_split = re.split("(?<=\\]) \\+ ",element[rated_data_full])
                this_rated_simple_split = element[rated_data_simple].split(" + ")
                this_rating_data_split = [float('NaN')] if pd.isna(element[rating_data]) else [x.strip() for x in element[rating_data].split(" + ")]
            if mode == 'geschichtsauffassung':
                this_rated_full_split = element[rated_data_full].split(" + ")
                this_rated_simple_split = element[rated_data_simple].split(" + ")
                this_rating_data_split = [float('NaN')] if pd.isna(element[rating_data]) else [x.strip() for x in element[rating_data].split(" + ")]

            for j in range(len(this_rated_full_split)):
                author_list.append(element.author)
                title_list.append(element.title)
                full_list.append(this_rated_full_split[j])
                type_list.append(this_rated_simple_split[j])
                rating_list.append(this_rating_data_split[j])

    rating_table = pd.DataFrame({
        'author': author_list,
        'title': title_list,
        'full': full_list,
        'type': type_list,
        'rating': rating_list
    })
    return rating_table

def binarize_meta (meta):
    meta_bin = meta.copy()
    ratings_themes = get_rating_table(meta = meta_bin, mode = 'themes')
    ratings_entity = get_rating_table(meta = meta_bin, mode = 'entity')
    ratings_themes['author_title'] = ratings_themes['author'] + ' – ' + ratings_themes['title']
    ratings_entity['author_title'] = ratings_entity['author'] + ' – ' + ratings_entity['title']
    
    meta_bin['words'] = [len(' '.join(x).split(" ")) if str(x) != 'None' else float('NaN') for x in meta_bin['text_bestocr']]
    
    meta_bin['ballade'] = [1 if 'Ballade' in str(x) else 0 for x in meta['gattung']]
    meta_bin['sonett'] = [1 if 'Sonett' in str(x) else 0 for x in meta['gattung']]
    meta_bin['lied'] = [1 if 'Lied' in str(x) else 0 for x in meta['gattung']]
    meta_bin['rollengedicht'] = [1 if 'Rollengedicht' in str(x) else 0 for x in meta['gattung']]
    meta_bin['denkmal'] = [1 if 'Denkmal' in str(x) else 0 for x in meta['gattung']]
    meta_bin['nogenre'] = [1 if x == None else 0 for x in meta['gattung']]

    meta_bin['sprechakte_count'] = [x.count(" + ") + 1 for x in meta['sprechakte']]
    meta_bin['sprechakt_erzaehlen_vorhanden'] = [1 if 'Erzählen' in str(x) else 0 for x in meta['sprechakte']]
    meta_bin['sprechakt_beschreiben_vorhanden'] = [1 if 'Beschreiben' in str(x) else 0 for x in meta['sprechakte']]
    meta_bin['sprechakt_behaupten_vorhanden'] = [1 if 'Behaupten' in str(x) else 0 for x in meta['sprechakte']]
    meta_bin['sprechakt_auffordern_vorhanden'] = [1 if 'Auffordern' in str(x) else 0 for x in meta['sprechakte']]
    meta_bin['sprechakt_fragen_vorhanden'] = [1 if 'Fragen' in str(x) else 0 for x in meta['sprechakte']]

    meta_bin['praesens_vorhanden'] = [1 if 'Präsens' in str(x) else 0 for x in meta['tempus']]
    meta_bin['praeteritum_vorhanden'] = [1 if 'Präteritum' in str(x) else 0 for x in meta['tempus']]
    meta_bin['futur_vorhanden'] = [1 if 'Futur' in str(x) else 0 for x in meta['tempus']]
    meta_bin['praesens_praeteritum_vorhanden'] = [1 if 'Präteritum' in str(x) and 'Präsens' in str(x) else 0 for x in meta['tempus']]

    meta_bin['sprechinstanz_nicht_in_vergangenheit'] = [1 if x == 0 else 0 for x in meta['sprechinstanz_in_vergangenheit']]
    meta_bin['sprechinstanz_in_vergangenheit'] = [1 if x == 1 else 0 for x in meta['sprechinstanz_in_vergangenheit']]

    meta_bin['in_hohem_mass_konkret'] = [1 if x == 1 else 0 for x in meta['konkretheit']]

    meta_bin['gegenwartsdominant'] = [1 if x == 0 else 0 for x in meta['vergangenheitsdominant']]
    meta_bin['zeit_mitte'] = (meta['beginn'] + meta['ende']) / 2
    meta_bin['antike'] = [1 if x <= 499 else 0 for x in meta_bin['zeit_mitte']]
    meta_bin['mittelalter'] = [1 if 500 <= x <= 1499 else 0 for x in meta_bin['zeit_mitte']]
    meta_bin['neuzeit'] = [1 if x >= 1500 else 0 for x in meta_bin['zeit_mitte']]

    meta_bin['mittelraum_count'] = [0 if pd.isna(x) else x.count(" + ") + 1 for x in meta['mittelraum']]    
    meta_bin['kleinraum_count'] = [0 if pd.isna(x) else x.count(" + ") + 1 for x in meta['kleinraum']]
    deutscher_raum = ['Germanien', 'Fränkisches Reich', 'Ostfränkisches Reich', 'Heiliges Römisches Reich', 'eutsch']
    meta_bin['behandelt_deutschen_mittelraum'] = [1 if any(string in str(x) for string in deutscher_raum) else 0 for x in meta['mittelraum']]
    meta_bin['behandelt_aussereuropa'] = [1 if pd.notna(x) and x != 'Europa' else 0 for x in meta['grossraum']]
    
    meta_bin['entity_count'] = [x.count(" + ") + 1 for x in meta['entity_simple']]
    meta_bin['entity_neutral'] = [x.count("0") for x in meta['entity_bewertung']]
    meta_bin['entity_positiv'] = [x.count("1") for x in meta['entity_bewertung']]
    meta_bin['entity_negativ'] = [x.count("2") for x in meta['entity_bewertung']]
    meta_bin['entity_ambivalent'] = [x.count("3") for x in meta['entity_bewertung']]
    
    meta_bin['bekanntes_individuum_count'] = [x.count("1") for x in meta['entity_simple']]
    b_ind_titles = ratings_entity.query("type == '1'")['author_title'].tolist()
    b_ind_positiv_titles = ratings_entity.query("type == '1' and rating == '1'")['author_title'].tolist()
    meta_bin['bekanntes_individuum_positiv'] = [1 if x in b_ind_positiv_titles else 0 if x in b_ind_titles else float('NaN') for x in meta_bin['author_title']]
    b_ind_negativ_titles = ratings_entity.query("type == '1' and rating == '2'")['author_title'].tolist()
    meta_bin['bekanntes_individuum_negativ'] = [1 if x in b_ind_negativ_titles else 0 if x in b_ind_titles else float('NaN') for x in meta_bin['author_title']]
    
    meta_bin['unbekanntes_individuum_count'] = [x.count("2") for x in meta['entity_simple']]
    ub_ind_titles = ratings_entity.query("type == '2'")['author_title'].tolist()
    ub_ind_positiv_titles = ratings_entity.query("type == '2' and rating == '1'")['author_title'].tolist()
    meta_bin['unbekanntes_individuum_positiv'] = [1 if x in ub_ind_positiv_titles else 0 if x in ub_ind_titles else float('NaN') for x in meta_bin['author_title']]
    ub_ind_negativ_titles = ratings_entity.query("type == '2' and rating == '2'")['author_title'].tolist()
    meta_bin['unbekanntes_individuum_negativ'] = [1 if x in ub_ind_negativ_titles else 0 if x in ub_ind_titles else float('NaN') for x in meta_bin['author_title']]
       
    meta_bin['kollektiv_count'] = [x.count("3") for x in meta['entity_simple']]  
    kollektiv_titles = ratings_entity.query("type == '3'")['author_title'].tolist()
    kollektiv_positiv_titles = ratings_entity.query("type == '3' and rating == '1'")['author_title'].tolist()
    meta_bin['kollektiv_positiv'] = [1 if x in kollektiv_positiv_titles else 0 if x in kollektiv_titles else float('NaN') for x in meta_bin['author_title']]
    kollektiv_negativ_titles = ratings_entity.query("type == '3' and rating == '2'")['author_title'].tolist()
    meta_bin['kollektiv_negativ'] = [1 if x in kollektiv_negativ_titles else 0 if x in kollektiv_titles else float('NaN') for x in meta_bin['author_title']]
           
    meta_bin['nichtmensch_count'] = [x.count("4") for x in meta['entity_simple']]
    
    meta_bin['ereignis'] = [1 if 'Ereignis' in x else 0 for x in meta['inhaltstyp']]
    meta_bin['zustand'] = [1 if 'Zustand' in x else 0 for x in meta['inhaltstyp']]

    meta_bin['stoffgebiet_count'] = [x.count(" + ") + 1 for x in meta['stoffgebiet']]
    meta_bin['stoffgebiet_neutral'] = [x.count("0") for x in meta['stoffgebiet_bewertung']]
    meta_bin['stoffgebiet_positiv'] = [x.count("1") for x in meta['stoffgebiet_bewertung']]
    meta_bin['stoffgebiet_negativ'] = [x.count("2") for x in meta['stoffgebiet_bewertung']]
    meta_bin['stoffgebiet_ambivalent'] = [x.count("3") for x in meta['stoffgebiet_bewertung']]
        
    meta_bin['krieg'] = [1 if 'Krieg' in x else 0 for x in meta['stoffgebiet']]    
    krieg_titles = ratings_themes.query("type == 'Militär/Krieg'")['author_title'].tolist()
    krieg_positiv_titles = ratings_themes.query("type == 'Militär/Krieg' and rating == '1'")['author_title'].tolist()
    meta_bin['krieg_positiv'] = [1 if x in krieg_positiv_titles else 0 if x in krieg_titles else float('NaN') for x in meta_bin['author_title']]
    krieg_negativ_titles = ratings_themes.query("type == 'Militär/Krieg' and rating == '2'")['author_title'].tolist()
    meta_bin['krieg_negativ'] = [1 if x in krieg_negativ_titles else 0 if x in krieg_titles else float('NaN') for x in meta_bin['author_title']]

    meta_bin['politik'] = [1 if 'Politik' in x else 0 for x in meta['stoffgebiet']]
    politik_titles = ratings_themes.query("type == 'Politik'")['author_title'].tolist()
    politik_positiv_titles = ratings_themes.query("type == 'Politik' and rating == '1'")['author_title'].tolist()
    meta_bin['politik_positiv'] = [1 if x in politik_positiv_titles else 0 if x in politik_titles else float('NaN') for x in meta_bin['author_title']]
    politik_negativ_titles = ratings_themes.query("type == 'Politik' and rating == '2'")['author_title'].tolist()
    meta_bin['politik_negativ'] = [1 if x in politik_negativ_titles else 0 if x in politik_titles else float('NaN') for x in meta_bin['author_title']]

    meta_bin['religion'] = [1 if 'Religion' in x else 0 for x in meta['stoffgebiet']]
    religion_titles = ratings_themes.query("type == 'Religion'")['author_title'].tolist()
    religion_positiv_titles = ratings_themes.query("type == 'Religion' and rating == '1'")['author_title'].tolist()
    meta_bin['religion_positiv'] = [1 if x in religion_positiv_titles else 0 if x in religion_titles else float('NaN') for x in meta_bin['author_title']]
    religion_negativ_titles = ratings_themes.query("type == 'Religion' and rating == '2'")['author_title'].tolist()
    meta_bin['religion_negativ'] = [1 if x in religion_negativ_titles else 0 if x in religion_titles else float('NaN') for x in meta_bin['author_title']]

    meta_bin['tod'] = [1 if 'Tod' in x else 0 for x in meta['stoffgebiet']]
    tod_titles = ratings_themes.query("type == 'Tod'")['author_title'].tolist()
    tod_positiv_titles = ratings_themes.query("type == 'Tod' and rating == '1'")['author_title'].tolist()
    meta_bin['tod_positiv'] = [1 if x in tod_positiv_titles else 0 if x in tod_titles else float('NaN') for x in meta_bin['author_title']]
    tod_negativ_titles = ratings_themes.query("type == 'Tod' and rating == '2'")['author_title'].tolist()
    meta_bin['tod_negativ'] = [1 if x in tod_negativ_titles else 0 if x in tod_titles else float('NaN') for x in meta_bin['author_title']]

    meta_bin['liebe'] = [1 if 'Liebe' in x else 0 for x in meta['stoffgebiet']]
    liebe_titles = ratings_themes.query("type == 'Liebe'")['author_title'].tolist()
    liebe_positiv_titles = ratings_themes.query("type == 'Liebe' and rating == '1'")['author_title'].tolist()
    meta_bin['liebe_positiv'] = [1 if x in liebe_positiv_titles else 0 if x in liebe_titles else float('NaN') for x in meta_bin['author_title']]
    liebe_negativ_titles = ratings_themes.query("type == 'Liebe' and rating == '2'")['author_title'].tolist()
    meta_bin['liebe_negativ'] = [1 if x in liebe_negativ_titles else 0 if x in liebe_titles else float('NaN') for x in meta_bin['author_title']]
    
    meta_bin['nation_volk_d'] = [1 if 'Nation/Volk-D' in x else 0 for x in meta['stoffgebiet']]
    nation_titles = ratings_themes.query("type == 'Nation/Volk-D'")['author_title'].tolist()
    nation_positiv_titles = ratings_themes.query("type == 'Nation/Volk-D' and rating == '1'")['author_title'].tolist()
    meta_bin['nation_volk_d_positiv'] = [1 if x in nation_positiv_titles else 0 if x in nation_titles else float('NaN') for x in meta_bin['author_title']]
    nation_negativ_titles = ratings_themes.query("type == 'Nation/Volk-D' and rating == '2'")['author_title'].tolist()
    meta_bin['nation_volk_d_negativ'] = [1 if x in nation_negativ_titles else 0 if x in nation_titles else float('NaN') for x in meta_bin['author_title']]

    meta_bin['marker_count'] = [(x[['marker_person', 'marker_zeit', 'marker_ort', 'marker_objekt']] != '/').sum() for x in meta.iloc]
    meta_bin['persmarker_vorhanden'] = [1 if x != '/' else 0 for x in meta['marker_person']]
    meta_bin['zeitmarker_vorhanden'] = [1 if x != '/' else 0 for x in meta['marker_zeit']]
    meta_bin['ortmarker_vorhanden'] = [1 if x != '/' else 0 for x in meta['marker_ort']]
    meta_bin['objektmarker_vorhanden'] = [1 if x != '/' else 0 for x in meta['marker_objekt']]

    meta_bin['ueberlieferung_positiv'] = [1 if x == 'positiv' else 0 if pd.notna(x) else x for x in meta['ueberlieferung_bewertung']]
    meta_bin['ueberlieferung_negativ'] = [1 if x == 'negativ' else 0 if pd.notna(x) else x for x in meta['ueberlieferung_bewertung']]
    meta_bin['geschichtsauffassung_positiv'] = [1 if x == 'positiv' else 0 if pd.notna(x) else x for x in meta['geschichtsauffassung_bewertung']]
    meta_bin['geschichtsauffassung_negativ'] = [1 if x == 'negativ' else 0 if pd.notna(x) else x for x in meta['geschichtsauffassung_bewertung']]

    meta_bin['wissen_ergaenzend'] = [1 if 'ergänzend' in str(x) else 0 for x in meta['verhaeltnis_wissen']]
    meta_bin['wissen_identisch'] = [1 if 'übereinstimmend' in str(x) else 0 for x in meta['verhaeltnis_wissen']]
    meta_bin['unwissend'] = [1 if x == -1 else 0 for x in meta['wissen']]
    meta_bin['wissend'] = [1 if x == 1 else 0 for x in meta['wissen']]
    meta_bin['reim'] = [1 if x == 1 else 0 for x in meta['reim']]
    meta_bin['metrum'] = [1 if x == 1 else 0 for x in meta['metrum']]
    meta_bin['verfremdung'] = [1 if x > 0 else 0 for x in meta['verfremdung']]
    
    meta_bin = meta_bin.drop([
        'sprechinstanz_zeitebene', 
        'vergangenheitsdominant', 
        # 'beginn', 'ende',
        'marker_person',
        'marker_zeit',
        'marker_ort',
        'marker_objekt',
        # 'konkretheit',
    ], axis = 1, errors='ignore')
    
    meta_bin = meta_bin.drop([
        'anthology_year_first_ed', 
        'anthology_year_used_ed', 
        'author_birth',
        'author_death', 
        'author_gnd_available',
        'author_gnd_gender',
        'author_gnd_birth',
        'author_gnd_death',
        'written_gt',
        'published_gt',
        'year_gt', 
        'year_predict_mean', 
        'year_predict_window_median_smoothed', 
        'year_predict_rfr', 
        'digitized', 'ocr_accuracy', 'ocr_column_error', 
        'ocr_accuracy_bestocr', 'ocr_column_error_bestocr',
    ], axis = 1, errors='ignore')

    meta_bin = meta_bin.reset_index(drop = True)
    
    return meta_bin
    
feature_dict = {
    'geschichtslyrik' : 'bin',
    'empirisch' : 'bin',
    'theoretisch' : 'bin',
    'gattung' : 'nom_multi',
    'sprechinstanz_markiert' : 'bin',
    'sprechakte' : 'nom_multi',
    'tempus' : 'nom_multi',
    'konkretheit' : 'cont',
    'wissen' : 'cont',
    'zeitebenen' : 'cont',
    'fixierbarkeit' : 'bin',
    'beginn' : 'cont',
    'ende' : 'cont',
    'anachronismus' : 'bin',
    'gegenwartsbezug' : 'bin',
    'grossraum' : 'nom_multi',
    'mittelraum' : 'nom_multi',
    'kleinraum' : 'nom_multi',
    'inhaltstyp' : 'nom_multi',
    'stoffgebiet' : 'nom_multi',
    'stoffgebiet_bewertung' : 'dependent',
    'entity_full' : 'nom_multi',
    'entity_simple' : 'nom_multi',
    'entity_bewertung' : 'dependent',
    'nationalismus' : 'bin',
    'heroismus' : 'bin',
    'religiositaet' : 'bin',
    'ueberlieferung' : 'bin',
    'ueberlieferung_bewertung' : 'dependent',
    'geschichtsauffassung' : 'bin',
    'geschichtsauffassung_bewertung' : 'dependent',
    'verhaeltnis_wissen' : 'nom_multi',
    'reim' : 'bin',
    'metrum' : 'bin',
    'verfremdung' : 'bin',
    'words' : 'cont',
    'ballade' : 'bin',
    'sonett' : 'bin',
    'lied' : 'bin',
    'rollengedicht' : 'bin',
    'denkmal' : 'bin',
    'nogenre' : 'bin',
    'sprechakte_count' : 'cont',
    'sprechakt_erzaehlen_vorhanden' : 'bin',
    'sprechakt_beschreiben_vorhanden' : 'bin',
    'sprechakt_behaupten_vorhanden' : 'bin',
    'sprechakt_auffordern_vorhanden' : 'bin',
    'sprechakt_fragen_vorhanden' : 'bin',
    'praesens_vorhanden' : 'bin',
    'praeteritum_vorhanden' : 'bin',
    'futur_vorhanden' : 'bin',
    'praesens_praeteritum_vorhanden' : 'bin',
    'sprechinstanz_nicht_in_vergangenheit' : 'bin',
    'sprechinstanz_in_vergangenheit' : 'bin',
    'in_hohem_mass_konkret' : 'bin',
    'gegenwartsdominant' : 'bin',
    'zeit_mitte' : 'cont',
    'antike' : 'bin',
    'mittelalter' : 'bin',
    'neuzeit' : 'bin',
    'mittelraum_count' : 'cont',
    'kleinraum_count' : 'cont',
    'behandelt_deutschen_mittelraum' : 'bin',
    'behandelt_aussereuropa' : 'bin',
    'entity_count' : 'cont',
    'entity_neutral' : 'bin',
    'entity_positiv' : 'bin',
    'entity_negativ' : 'bin',
    'entity_ambivalent' : 'bin',
    'bekanntes_individuum_count' : 'cont',
    'bekanntes_individuum_positiv' : 'bin',
    'bekanntes_individuum_negativ' : 'bin',
    'unbekanntes_individuum_count' : 'cont',
    'unbekanntes_individuum_positiv' : 'bin',
    'unbekanntes_individuum_negativ' : 'bin',
    'kollektiv_count' : 'cont',
    'kollektiv_positiv' : 'bin',
    'kollektiv_negativ' : 'bin',
    'nichtmensch_count' : 'cont',
    'ereignis' : 'bin',
    'zustand' : 'bin',
    'stoffgebiet_count' : 'cont',
    'stoffgebiet_neutral' : 'bin',
    'stoffgebiet_positiv' : 'bin',
    'stoffgebiet_negativ' : 'bin',
    'stoffgebiet_ambivalent' : 'bin',
    'krieg' : 'bin',
    'krieg_positiv' : 'bin',
    'krieg_negativ' : 'bin',
    'politik' : 'bin',
    'politik_positiv' : 'bin',
    'politik_negativ' : 'bin',
    'religion' : 'bin',
    'religion_positiv' : 'bin',
    'religion_negativ' : 'bin',
    'tod' : 'bin',
    'tod_positiv' : 'bin',
    'tod_negativ' : 'bin',
    'liebe' : 'bin',
    'liebe_positiv' : 'bin',
    'liebe_negativ' : 'bin',
    'nation_volk_d' : 'bin',
    'nation_volk_d_positiv' : 'bin',
    'nation_volk_d_negativ' : 'bin',
    'marker_count' : 'cont',
    'persmarker_vorhanden' : 'bin',
    'zeitmarker_vorhanden' : 'bin',
    'ortmarker_vorhanden' : 'bin',
    'objektmarker_vorhanden' : 'bin',
    'ueberlieferung_positiv' : 'bin',
    'ueberlieferung_negativ' : 'bin',
    'geschichtsauffassung_positiv' : 'bin',
    'geschichtsauffassung_negativ' : 'bin',
    'wissen_ergaenzend' : 'bin',
    'wissen_identisch' : 'bin',
    'unwissend' : 'bin',
    'wissend' : 'bin',
    'jahrhundert_mitte' : 'cont',
    'zeit_mitte' : 'cont',
    'dekade_mitte' : 'cont',
	'wissen_behandelt' : 'bin',
	'count' : 'cont',
}

def create_ts_plot (data, columns, y_axis_title, add_corporas=[], add_corpora_names=[], add_corpora_categories=[], confint_columns=[]):
    fig = go.Figure()

    # confint
    if len(confint_columns)>0:
        for i, confint_column in enumerate(confint_columns):
            fig.add_trace(
                go.Scatter(
                    x=data.index.tolist(),
                    y=ts[confint_column[0]].tolist(),
                    mode='lines',
                    line=dict(width=0, color=px.colors.qualitative.Plotly[i]),
                    showlegend=False
                ))
            fig.add_trace(
                go.Scatter(
                    x=data.index.tolist(),
                    y=ts[confint_column[1]].tolist(),
                    line=dict(width=0, color=px.colors.qualitative.Plotly[i]),
                    mode='lines',
                    fillcolor=plotly_colors_rgba[i],
                    fill='tonexty',
                    showlegend=False
                )
            )

    # regular
    for i, column in enumerate(data.columns):
        fig.add_trace(
            go.Scatter(
                name = column,
                x=data.index.tolist(),
                y=data[column].tolist(),
                mode='lines',
                line=dict(color = px.colors.qualitative.Plotly[i], width=6),
            )
        )

    # Ergänzungskorpora
    if len(add_corporas)>0:
        for i, this_name in enumerate([x for x in add_corpora_names if x != 'Anthologien']):
            fig.add_trace(
                go.Scatter(
                    name = f"{this_name}",
                    x = [add_corporas.loc[this_name, 'Jahr']]*4,
                    y = add_corporas.loc[this_name, add_corpora_categories],
                    mode = 'markers',
                    marker_size = 14,
                    marker_symbol = SymbolValidator().values[2 + 12*i],
                    marker_color = px.colors.qualitative.Plotly[:6],
                    showlegend=False
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    name=f"{this_name}",
                    x=[None], y=[None],
                    mode='markers',
                    marker_size=14,
                    marker_symbol = SymbolValidator().values[2 + 12*i],
                    marker_color = 'black',            
                )
            )
    
    #layout
    fig.update_layout(
        width=900, height=500,
        xaxis=dict(tickfont=dict(size=16), titlefont=dict(size=16)),
        yaxis=dict(title=y_axis_title, tickfont=dict(size=16), titlefont=dict(size=16)),
        legend=dict(font = dict(size=16), traceorder = 'normal'),
    )

    return fig

def save_ts_data (data, prefix='NoPrefixSpecified_'):
    ts_results = pd.read_csv("../resources/more/time_series_results.csv", index_col=0)
    for col in data.columns:
        ts_results[prefix+col] = data[col].dropna()
    ts_results.to_csv("../resources/more/time_series_results.csv")
    
def update_fig_for_publication(fig, make_grey = False):
    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        font_color="black",
        xaxis=dict(
            color="black",
            gridcolor="white",  # no vertical gridlines
            showgrid=False
        ),
        yaxis=dict(
            color="black",
            gridcolor="lightgray",  # horizontal gridlines
            zerolinecolor="lightgray"
        ),
        legend=dict(
            bordercolor="black",
            borderwidth=1  # Adjust thickness as needed
        )
    )
    if make_grey:
        for trace in fig.data:
            if 'marker' in trace:
                trace.marker.color = "#555555"
            if 'line' in trace:
                trace.line.color = "#555555"
    return fig
