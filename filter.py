import json


def filter_tracks(jsonl_path, artist_name=None, danceability_range=None, energy_range=None, release_year_range=None, docid_list=None):
    """
    Filters tracks from a JSONL file based on specified criteria, restricted to a provided list of doc IDs.
    """
    if docid_list is None:
        raise ValueError("docid_list must be provided to restrict filtering.")

    docid_set = set(docid_list)
    filtered_tracks = []

    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            track = json.loads(line)
            if track['docid'] not in docid_set:
                continue
            if artist_name and artist_name.lower() not in track.get('artists', '').lower():
                continue
            if danceability_range:
                min_danceability, max_danceability = danceability_range
                if not (min_danceability <= track.get('danceability', 0) <= max_danceability):
                    continue
            if energy_range:
                min_energy, max_energy = energy_range
                if not (min_energy <= track.get('energy', 0) <= max_energy):
                    continue
            if release_year_range:
                min_year, max_year = release_year_range
                release_year = int(track.get('release_date', '1900').split('-')[0])
                if not (min_year <= release_year <= max_year):
                    continue

            filtered_tracks.append(track)

    return filtered_tracks


