import joblib
from tqdm import tqdm
from Source.Miners.SimpleParser import *
import pandas as pd
from datetime import timedelta
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Source.Config import *


account_list_df = pd.read_csv(account_list_filepath)
account_list_df['registered_on'] = pd.to_datetime(account_list_df['registered_on'])

change_list_df = joblib.load(selected_change_list_filepath)
change_list_df = change_list_df.sort_values(by=['change_id']).reset_index(drop=True)
for col in ['created', 'updated']:
    change_list_df.loc[:, col] = change_list_df[col].apply(pd.to_datetime)

lookback = 120
file_feature_maps = {}


def main():
    output_file_name = f"{root}/{project}_fan_fixed.csv"

    features_list = [
        'change_num', 'recent_change_num', 'subsystem_change_num', 'review_num', 'merged_ratio',
        'recent_merged_ratio', 'subsystem_merged_ratio',

        'lines_added_num', 'lines_deleted_num', 'changed_file_num', 'files_added_num', 'files_deleted_num',
        'directory_num', 'subsystem_num', 'modify_entropy', 'language_num', 'file_type_num', 'segs_added_num',
        'segs_deleted_num', 'segs_updated_num',

        'changes_files_modified', 'file_developer_num',

        'degree_centrality', 'closeness_centrality', 'betweenness_centrality',
        'eigenvector_centrality', 'clustering_coefficient', 'k_coreness',

        'msg_length', 'has_bug', 'has_feature', 'has_improve', 'has_document', 'has_refactor'
    ]
    file_header = ["project", "change_id", 'created'] + features_list + ['status']

    initialize(output_file_name, file_header)
    csv_file = open(output_file_name, "a", newline='', encoding='utf-8')
    file_writer = csv.writer(csv_file, dialect='excel')

    change_ids = change_list_df['change_id'].values

    # it is important to calculate in sorted order of created.
    # Change numbers are given in increasing order of creation time
    count = 0
    for change_id in tqdm(change_ids):
        # print(change_number)
        filename = f'{project}_{change_id}_change.json'
        filepath = os.path.join(changes_root, filename)
        if not os.path.exists(filepath):
            print(f'{filename} does not exist')
            continue

        change = Change(json.load(open(filepath, "r")))
        if not change.is_real_change():
            continue

        current_date = pd.to_datetime(change.first_revision.created)
        calculator = FeatureCalculator(change, current_date)

        author_features = calculator.owner_features_fixed
        code_features = calculator.code_features
        file_history_features = calculator.file_history_features

        text_features = calculator.text_features
        social_features = calculator.social_features

        status = 1 if change.status == 'MERGED' else 0

        feature_vector = [
            change.project, change_id, change.created,

            author_features['change_num'], author_features['recent_change_num'],
            author_features['subsystem_change_num'],
            author_features['review_num'], author_features['merged_ratio'],
            author_features['recent_merged_ratio'], author_features['subsystem_merged_ratio'],

            code_features['lines_added_num'], code_features['lines_deleted_num'], code_features['changed_file_num'],
            code_features['file_added_num'], code_features['file_deleted_num'], code_features['directory_num'],
            code_features['subsystem_num'], code_features['modify_entropy'], code_features['language_num'],
            code_features['file_type_num'], code_features['segs_added_num'], code_features['segs_deleted_num'],
            code_features['segs_updated_num'],

            file_history_features['changes_files_modified'], file_history_features['file_developer_num'],

            social_features['degree_centrality'], social_features['closeness_centrality'],
            social_features['betweenness_centrality'], social_features['eigenvector_centrality'],
            social_features['clustering_coefficient'], social_features['k_coreness'],

            text_features['msg_length'], text_features['has_bug'], text_features['has_feature'],
            text_features['has_improve'], text_features['has_document'], text_features['has_refactor'],

            status
        ]
        file_writer.writerow(feature_vector)

        count += 1
        if count % 100 == 0:
            csv_file.flush()
            # break

    csv_file.close()

    features = pd.read_csv(output_file_name)
    features.drop_duplicates(['change_id'], inplace=True)
    features.to_csv(output_file_name, index=False, float_format='%.2f')


def initialize(path, file_header):
    with open(path, "w", newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', dialect='excel')
        writer.writerow(file_header)
        csvfile.close()


class SocialNetwork:
    def __init__(self, graph, owner):
        self.graph = graph
        self.owner = owner
        self.lcc = self.largest_connected_component()

    def show_graph(self):
        nx.draw(self.graph, with_labels=True, font_weight='bold')
        plt.show()

    def largest_connected_component(self):
        try:
            return self.graph.subgraph(max(nx.connected_components(self.graph), key=len))
        except:
            return self.graph

    def degree_centrality(self):
        nodes_dict = nx.degree_centrality(self.lcc)
        try:
            return nodes_dict[self.owner]
        except:
            return 0

    def closeness_centrality(self):
        try:
            return nx.closeness_centrality(self.lcc, u=self.owner)
        except:
            return 0

    def betweenness_centrality(self):
        nodes_dict = nx.betweenness_centrality(self.lcc, weight='weight')
        try:
            return nodes_dict[self.owner]
        except:
            return 0

    def eigenvector_centrality(self):
        try:
            eigenvector_centrality = nx.eigenvector_centrality(self.lcc)
            try:
                return eigenvector_centrality[self.owner]
            except:
                return 0
        except:
            return 0

    def clustering_coefficient(self):
        try:
            return nx.clustering(self.lcc, nodes=self.owner, weight='weight')
        except:
            return 0

    def k_coreness(self):
        nodes_dict = nx.core_number(self.lcc)
        try:
            return nodes_dict[self.owner]
        except:
            return 0


class FeatureCalculator:
    def __init__(self, change, current_date):
        self.change = change
        self.project = change.project
        self.current_date = current_date
        self.old_date = current_date - timedelta(days=lookback)

    @property
    def owner_features_fixed(self):
        features, owner = {}, self.change.owner
        prev_works = change_list_df[change_list_df['created'] < self.current_date]
        features['review_num'] = prev_works[prev_works['reviewers'].apply(lambda x: owner in x)].shape[0]

        author_works = prev_works[prev_works['owner'] == owner]
        features['change_num'] = author_works.shape[0]

        finished_works = author_works[author_works['updated'] <= self.current_date]
        merged_works = finished_works[finished_works['status'] == 'MERGED']
        features['merged_ratio'] = float(merged_works.shape[0]) / max(finished_works.shape[0], 1)

        recent_change_num = 0
        closed_recent_change_num = 0
        recent_change_merged = 0
        subsys_changes = set()
        closed_subsys_changes = set()
        merged_subsys_changes = set()
        current_subsystems = self.change.subsystems

        for (_, change_number, _, _, created, updated, _, _, subsystems, status) in author_works.itertuples(name=None):
            delta_create_day = (self.current_date - created).days

            if delta_create_day <= lookback:
                recent_change_num += 1.0 / (float(delta_create_day) / 30 + 1)
                # if the change has been closed by now
                if updated <= self.current_date:
                    closed_recent_change_num += 1.0 / (float(delta_create_day) / 30 + 1)
                    if status == 'MERGED':
                        recent_change_merged += 1.0 / (float(delta_create_day) / 30 + 1)

            for subsystem in current_subsystems:
                if subsystem in subsystems:
                    subsys_changes.add(change_number)
                    if updated <= self.current_date:
                        closed_subsys_changes.add(change_number)
                        if status == 'MERGED':
                            merged_subsys_changes.add(change_number)
                    break

        features['recent_change_num'] = recent_change_num
        if closed_recent_change_num != 0:
            features['recent_merged_ratio'] = float(recent_change_merged) / closed_recent_change_num
        else:
            features['recent_merged_ratio'] = 0

        features['subsystem_change_num'] = len(subsys_changes)
        features['subsystem_merged_ratio'] = float(len(merged_subsys_changes)) / max(len(closed_subsys_changes), 1)

        return features

    @property
    def owner_features_original(self):
        features, owner = {}, self.change.owner
        prev_works = change_list_df[change_list_df['created'] <= self.current_date]
        features['review_num'] = prev_works[prev_works['reviewers'].apply(lambda x: owner in x)].shape[0]

        author_works = prev_works[prev_works['owner'] == owner]
        features['change_num'] = author_works.shape[0]

        merged_works = author_works[author_works['status'] == 'MERGED']
        # some of these changes are not finished yet. so this contains future data
        features['merged_ratio'] = float(merged_works.shape[0]) / max(author_works.shape[0], 1)

        recent_change_num = 0
        recent_change_merged = 0
        subsys_changes = set()
        merged_subsys_changes = set()
        current_subsystems = self.change.subsystems

        for (_, change_number, _, _, created, updated, _, _, subsystems, status) in author_works.itertuples(name=None):
            delta_create_day = (self.current_date - created).days

            # if the change is within last 120 days
            if delta_create_day <= lookback:
                recent_change_num += 1.0 / (float(delta_create_day) / 30 + 1)
                # not excluding open changes when calculating merge ratio.
                if status == 'MERGED':
                    recent_change_merged += 1.0 / (float(delta_create_day) / 30 + 1)

            for subsystem in current_subsystems:
                if subsystem in subsystems:
                    subsys_changes.add(change_number)
                    # not excluding open changes when calculating merge ratio
                    if status == 'MERGED':
                        merged_subsys_changes.add(change_number)
                    break

        features['recent_change_num'] = recent_change_num
        if recent_change_num != 0:
            features['recent_merged_ratio'] = float(recent_change_merged) / recent_change_num
        else:
            features['recent_merged_ratio'] = 0

        n_subsystem_merged = len(merged_subsys_changes)
        features['subsystem_change_num'] = len(subsys_changes)
        features['subsystem_merged_ratio'] = float(n_subsystem_merged) / max(len(subsys_changes), 1)

        return features

    @property
    def code_features(self):
        features, files = {}, self.change.files

        files_added = files_deleted = 0
        lines_added = lines_deleted = 0

        directories = set()
        subsystems = set()
        for file in files:
            lines_added += file.lines_inserted
            lines_deleted += file.lines_deleted

            if file.status == 'D': files_deleted += 1
            if file.status == 'A': files_added += 1

            names = file.path.split('/')
            if len(names) > 1:
                directories.update([names[-2]])
                subsystems.update(names[0])

        lines_changed = lines_added + lines_deleted
        if lines_added > 0:
            lines_added = np.log2(lines_added) + 1
        if lines_deleted:
            lines_deleted = np.log2(lines_deleted) + 1

        features['lines_added_num'] = lines_added
        features['lines_deleted_num'] = lines_deleted

        features['directory_num'] = len(directories)
        features['subsystem_num'] = len(subsystems)

        features['file_added_num'] = files_added
        features['file_deleted_num'] = files_deleted
        features['changed_file_num'] = len(files) - files_deleted - files_added

        # Entropy is defined as: −Sum(k=1 to n)(pk∗log2pk). Note that n is number of files
        # modified in the change, and pk is calculated as the proportion of lines modified in file k among
        # lines modified in this code change.
        modify_entropy = 0
        if lines_changed:
            for file in files:
                lines_changed_in_file = file.lines_deleted + file.lines_inserted
                if lines_changed_in_file:
                    pk = float(lines_changed_in_file) / lines_changed
                    modify_entropy -= pk * np.log2(pk)

        features['modify_entropy'] = modify_entropy
        features['file_type_num'] = self.change.file_type_num
        features['language_num'] = self.change.language_num

        filepath = os.path.join(diff_root, f"{project}_{self.change.change_number}_diff.json")
        diff_json = json.load(open(filepath, 'r'))

        segs_added = segs_deleted = segs_updated = 0

        try:
            files = list(diff_json.values())[0].values()
            for file in files:
                for content in file['content']:
                    change_type = list(content.keys())
                    if change_type == ['a']:
                        segs_deleted += 1
                    elif change_type == ['a', 'b']:
                        segs_updated += 1
                    elif change_type == ['b']:
                        segs_added += 1

        except IndexError:
            print('Error for {0}'.format(self.change.change_number))

        features['segs_added_num'] = segs_added
        features['segs_deleted_num'] = segs_deleted
        features['segs_updated_num'] = segs_updated

        return features

    @property
    def file_history_features(self):
        use_global_file_map = {}
        for file in self.change.files:
            use_global_file_map[file.path] = get_file_feature_map(self.project, file.path)

        modify_changes = set()
        developers = set()
        for file_path in use_global_file_map.keys():
            modify_changes |= use_global_file_map[file_path]['modify_changes']
            developers |= use_global_file_map[file_path]['developer']
            use_global_file_map[file_path]['modify_changes'].add(self.change.change_number)
            use_global_file_map[file_path]['developer'].add(self.change.owner)

        return {
            'changes_files_modified': len(modify_changes),
            'file_developer_num': len(developers)
        }

    @property
    def social_features(self):
        old_date = pd.to_datetime(self.change.created) - timedelta(days=30)
        df = change_list_df[change_list_df['project'] == self.project]
        df = df[(df['created'] >= old_date) & (df['created'] < self.change.created)]
        owners, reviewers_list = df['owner'].values, df['reviewers'].values

        graph = nx.Graph()
        for index in range(df.shape[0]):
            owner, reviewers = owners[index], reviewers_list[index]
            for reviewer in reviewers:
                if reviewer == owner: continue
                try:
                    graph[owner][reviewer]['weight'] += 1
                except (KeyError, IndexError):
                    graph.add_edge(owner, reviewer, weight=1)

        network = SocialNetwork(graph, self.change.owner)
        # network.show_graph()
        return {
            'degree_centrality': network.degree_centrality(),
            'closeness_centrality': network.closeness_centrality(),
            'betweenness_centrality': network.betweenness_centrality(),
            'eigenvector_centrality': network.eigenvector_centrality(),
            'clustering_coefficient': network.clustering_coefficient(),
            'k_coreness': network.k_coreness()
        }

    @property
    def text_features(self):
        if self.change.first_revision is None:
            return {}

        text_features = {}
        msg = self.change.subject.lower()
        text_features['msg'] = msg
        text_features['msg_length'] = len(msg.split())
        for keyword in ['bug', 'feature', 'improve', 'document', 'refactor']:
            text_features['has_' + keyword] = keyword in msg
        return text_features


def get_file_feature_map(sub_project_name, file_path):
    try:
        file_feature_maps[sub_project_name]
    except KeyError:
        file_feature_maps[sub_project_name] = {}

    try:
        file_feature_maps[sub_project_name][file_path]
    except KeyError:
        file_feature_maps[sub_project_name][file_path] = {'modify_changes': set(), 'developer': set()}
    return file_feature_maps[sub_project_name][file_path]


if __name__ == '__main__':
    main()