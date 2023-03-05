from datetime import timedelta

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import textstat
from tqdm import tqdm

from Source.Miners.SimpleParser import *
from Source.Util import *

account_list_df = pd.read_csv(account_list_filepath)
account_list_df['registered_on'] = pd.to_datetime(account_list_df['registered_on'])
account_list_df['name'] = account_list_df['name'].apply(str)

change_list_df = joblib.load(selected_change_list_filepath)
change_list_df = change_list_df.sort_values(by=['change_id']).reset_index(drop=True)
for col in ['created', 'updated', 'closed']:
    change_list_df.loc[:, col] = change_list_df[col].apply(pd.to_datetime)

comment_list_df = pd.read_csv(comment_list_filepath)
comment_list_df['updated'] = pd.to_datetime(comment_list_df['updated'])
comment_list_df['message'] = comment_list_df['message'].apply(str)

lookback = 60
default_changes = 1
default_merge_ratio = 0.5

# 计算历史PR的时间和轮次时，如果历史PR为空，则取默认值
default_duration = 1
default_rounds = 1


def main():
    features_list = initial_feature_list
    file_header = ["project", "change_id", 'created', 'subject'] + features_list + ['avg_score', 'status', 'time',
                                                                                    'rounds']
    output_file_name = f"{root}/{project}.csv"

    initialize(output_file_name, file_header)
    csv_file = open(output_file_name, "a", newline='', encoding='utf-8')
    file_writer = csv.writer(csv_file, dialect='excel')

    change_numbers = change_list_df['change_id'].values

    # it is important to calculate in sorted order of created.
    # Change numbers are given in increasing order of creation time
    count = 0
    # for change_number in change_numbers:
    for change_number in tqdm(change_numbers):
        # print(change_number)

        filename = f'{project}_{change_number}_change.json'
        filepath = os.path.join(changes_root, filename)
        if not os.path.exists(filepath):
            print(f'{filename} does not exist')
            continue

        change = Change(json.load(open(filepath, "r")))
        if not change.is_real_change():
            continue

        current_date = pd.to_datetime(change.first_revision.created)
        calculator = FeatureCalculator(change, current_date)

        author_features = calculator.author_features
        reviewer_features = calculator.reviewer_features
        file_features = calculator.file_features
        # file_diff_features = calculator.diff_features

        meta_features = calculator.meta_features

        project_features = calculator.project_features
        description_features = calculator.description_features

        social_features = calculator.social_features

        status = 1 if change.status == 'MERGED' else 0
        feature_vector = [
            change.project, change.change_number, change.created, change.subject,

            author_features['author_experience'], author_features['author_merge_ratio'],
            author_features['author_changes_per_week'], author_features['author_merge_ratio_in_project'],
            author_features['total_change_num'], author_features['author_review_num'],
            author_features['is_reviewer'],
            author_features['author_subsystem_change_num'], author_features['author_subsystem_change_merge_ratio'],
            author_features['author_avg_rounds'], author_features['author_contribution_rate'],
            author_features['author_merged_change_num'], author_features['author_abandoned_changes_num'],
            author_features['author_change_duration_min'], author_features['author_change_duration_max'],
            author_features['author_change_duration_avg'], author_features['author_change_duration_std'],

            description_features['description_length'], description_features['is_documentation'],
            description_features['is_bug_fixing'], description_features['is_feature'],
            description_features['is_improve'], description_features['is_refactor'],

            meta_features['commit_num'], meta_features['comment_num'], meta_features['comment_word_num'],
            meta_features['last_comment_mention'], meta_features['has_test_file'],
            meta_features['description_readability'],

            project_features['project_changes_per_week'], project_features['project_merge_ratio'],
            project_features['changes_per_author'],
            project_features['project_author_num'], project_features['project_duration_per_change'],
            project_features['project_commits_per_change'], project_features['project_comments_per_change'],
            project_features['project_file_num_per_change'], project_features['project_churn_per_change'],
            project_features['project_additions_per_week'], project_features['project_deletions_per_week'],
            project_features['workload'],

            reviewer_features['num_of_reviewers'], reviewer_features['num_of_bot_reviewers'],
            reviewer_features['avg_reviewer_experience'], reviewer_features['avg_reviewer_review_count'],
            reviewer_features['review_avg_rounds'], reviewer_features['review_avg_duration'],

            file_features['lines_added'], file_features['lines_updated'], file_features['lines_deleted'],
            file_features['modify_entropy'],
            file_features['files_added'], file_features['files_deleted'], file_features['files_modified'],
            file_features['num_of_directory'], file_features['subsystem_num'],
            file_features['language_num'], file_features['file_type_num'],
            file_features['segs_added'], file_features['segs_deleted'], file_features['segs_updated'],

            social_features['degree_centrality'], social_features['closeness_centrality'],
            social_features['betweenness_centrality'], social_features['eigenvector_centrality'],
            social_features['clustering_coefficient'], social_features['k_coreness'],

            meta_features['avg_score'],
            status,
            day_diff(change.closed, change.created),
            len(change.revisions)
        ]
        file_writer.writerow(feature_vector)

        count += 1
        if count % 100 == 0:
            csv_file.flush()
            # break

    csv_file.close()

    features = pd.read_csv(output_file_name)
    features.drop_duplicates(['change_id'], inplace=True)
    features.sort_values(by=['change_id']).to_csv(output_file_name, index=False, float_format='%.2f')


class FeatureCalculator:
    def __init__(self, change, current_date):
        self.change = change
        self.project = change.project
        self.current_date = current_date
        self.old_date = current_date - timedelta(days=lookback)

    @property
    def author_features(self):
        features, owner = {}, self.change.owner
        registered_on = account_list_df[account_list_df['account_id'] == owner]['registered_on'].values
        authors_work = change_list_df[
            (change_list_df['owner'] == owner) & (change_list_df['created'] < self.current_date)]
        features['total_change_num'] = authors_work.shape[0]
        # first_date = authors_work['created'].min()
        first_date = authors_work['created'].min() if authors_work.shape[0] > 0 else self.old_date

        if len(registered_on) == 0 or registered_on[0] > self.current_date:
            features['author_experience'] = max(0, day_diff(self.current_date, first_date) / 365.0)
        else:
            features['author_experience'] = day_diff(self.current_date, registered_on[0]) / 365.0

        ongoing_works = change_list_df[(self.old_date <= change_list_df['updated'])
                                       & (change_list_df['created'] <= self.current_date)]

        features['author_review_num'] = ongoing_works[ongoing_works['reviewers'].apply(lambda x: owner in x)].shape[0]

        finished_works = ongoing_works[
            (ongoing_works['owner'] == owner) & (ongoing_works['updated'] <= self.current_date)]
        merged_works = finished_works[finished_works['status'] == 'MERGED']

        if finished_works.shape[0] >= default_changes:
            features['author_merge_ratio'] = float(merged_works.shape[0]) / finished_works.shape[0]
        else:
            features['author_merge_ratio'] = default_merge_ratio

        weeks = max(day_diff(self.current_date, max(first_date, self.old_date)) / 7.0, 1)
        features['author_changes_per_week'] = finished_works.shape[0] / weeks
        if np.isnan(features['author_changes_per_week']):
            print(finished_works.shape[0])
            print(weeks)

        finished_changes_in_project = finished_works[finished_works['project'] == self.project].shape[0]
        if finished_changes_in_project >= default_changes:
            features['author_merge_ratio_in_project'] = float(
                merged_works[merged_works['project'] == self.project].shape[0]) / finished_changes_in_project
        else:
            features['author_merge_ratio_in_project'] = default_merge_ratio

        current_subsystem = self.change.subsystems
        subsystem_changes = authors_work[authors_work['subsystems'].apply(lambda x: len(x & current_subsystem) > 0)]
        features['author_subsystem_change_num'] = subsystem_changes.shape[0]
        features['author_subsystem_change_merge_ratio'] = \
            float(subsystem_changes[subsystem_changes['status'] == 'MERGED'].shape[0] / subsystem_changes.shape[0]) \
                if subsystem_changes.shape[0] > 0 else default_merge_ratio

        changes_until_current = change_list_df[(change_list_df['created'] < self.current_date)]

        # float(np.sum(authors_work['revision_num'].values) / authors_work.shape[0])\
        features['author_avg_rounds'] = np.mean(authors_work['revision_num'].values) \
            if authors_work.shape[0] > 0 else (np.mean(changes_until_current['revision_num'].values)
                                               if changes_until_current.shape[0] > 0 else default_rounds)

        features['author_contribution_rate'] = \
            float(authors_work.shape[0] / changes_until_current.shape[0]) if changes_until_current.shape[0] > 0 else 1.0

        features['is_reviewer'] = True if features['author_review_num'] > 0 else False

        features['author_merged_change_num'] = authors_work[authors_work['status'] == 'MERGED'].shape[0]
        features['author_abandoned_changes_num'] = authors_work[authors_work['status'] == 'ABANDONED'].shape[0]

        if authors_work.shape[0] > 0:
            features['author_change_duration_min'] = np.min(authors_work['duration'].values)
            features['author_change_duration_max'] = np.max(authors_work['duration'].values)
            features['author_change_duration_avg'] = np.mean(authors_work['duration'].values)
            features['author_change_duration_std'] = np.std(authors_work['duration'].values)
        elif changes_until_current.shape[0] > 0:
            features['author_change_duration_min'] = np.min(changes_until_current['duration'].values)
            features['author_change_duration_max'] = np.max(changes_until_current['duration'].values)
            features['author_change_duration_avg'] = np.mean(changes_until_current['duration'].values)
            features['author_change_duration_std'] = np.std(changes_until_current['duration'].values)
        else:
            features['author_change_duration_min'] = default_duration
            features['author_change_duration_max'] = default_duration
            features['author_change_duration_avg'] = default_duration
            features['author_change_duration_std'] = 0

        features['is_new_author'] = True if authors_work.shape[0] == 0 else False

        return features

    @property
    def project_features(self):
        features = {}

        works = change_list_df[change_list_df['project'] == self.project]
        finished_works = works[(self.old_date <= works['updated']) & (works['updated'] <= self.current_date)]

        if finished_works.shape[0] >= default_changes:
            features['project_merge_ratio'] = float(finished_works[finished_works['status'] == 'MERGED'].shape[0]) / \
                                              finished_works.shape[0]
        else:
            features['project_merge_ratio'] = default_merge_ratio

        # per week changes in the last lookback days
        features['project_changes_per_week'] = finished_works.shape[0] * 7.0 / lookback

        owner_num = finished_works['owner'].nunique()
        features['changes_per_author'] = 0
        if owner_num:
            features['changes_per_author'] = float(finished_works.shape[0]) / owner_num

        # 自己实现的
        features['project_author_num'] = owner_num

        changes_until_current = change_list_df[(change_list_df['updated'] < self.current_date)]
        if finished_works.shape[0] > 0:
            features['project_duration_per_change'] = np.mean(finished_works['duration'].values)
            features['project_commits_per_change'] = np.mean(finished_works['revision_num'].values)
            features['project_comments_per_change'] = np.mean(finished_works['comment_num'].values)
            features['project_file_num_per_change'] = np.mean(finished_works['file_num'].values)
            features['project_churn_per_change'] = np.mean(
                finished_works['added_lines'].values + finished_works['deleted_lines'].values)
        elif changes_until_current.shape[0] > 0:
            features['project_duration_per_change'] = np.mean(changes_until_current['duration'].values)
            features['project_commits_per_change'] = np.mean(changes_until_current['revision_num'].values)
            features['project_comments_per_change'] = np.mean(changes_until_current['comment_num'].values)
            features['project_file_num_per_change'] = np.mean(changes_until_current['file_num'].values)
            features['project_churn_per_change'] = np.mean(
                changes_until_current['added_lines'].values + changes_until_current['deleted_lines'].values)
        else:
            features['project_duration_per_change'] = 0
            features['project_commits_per_change'] = 0
            features['project_comments_per_change'] = 0
            features['project_file_num_per_change'] = 0
            features['project_churn_per_change'] = 0

        week_num = max(1, day_diff(self.current_date, self.old_date) / 7.0)
        features['project_additions_per_week'] = np.sum(finished_works['added_lines']) / week_num
        features['project_deletions_per_week'] = np.sum(finished_works['deleted_lines']) / week_num

        features['workload'] = \
            works[(works['created'] < self.current_date) & (self.current_date < works['closed'])].shape[0]

        return features

    @property
    def reviewer_features(self):
        features, reviewer_list = {}, self.change.reviewers
        ongoing_works = change_list_df[(self.old_date <= change_list_df['updated'])
                                       & (change_list_df['created'] <= self.current_date)]
        changes_until_current = change_list_df[(change_list_df['created'] < self.current_date)]

        avg_experience = avg_num_review = 0.0
        count = bot = 0
        # 所有评审人评审过的PR的id的集合
        related_change_ids = set()
        for reviewer_id in reviewer_list:
            result = account_list_df[account_list_df['account_id'] == reviewer_id]
            registered_on = result['registered_on'].values

            if len(registered_on) == 0 or self.current_date < registered_on[0]:
                continue

            if is_bot(self.project, result['name'].values[0]):
                bot += 1
                continue
            work_experience = day_diff(self.current_date, registered_on[0]) / 365.0  # convert to year
            avg_experience += work_experience

            avg_num_review += ongoing_works[ongoing_works['reviewers'].apply(lambda x: reviewer_id in x)].shape[0]
            count += 1
            # 将当前评审人评审过的所有PR的id加入related_change_ids集合
            if changes_until_current.shape[0] > 0:
                ralated_changes = changes_until_current[
                    changes_until_current['reviewers'].apply(lambda x: reviewer_id in x)]
                related_change_ids.update(ralated_changes['change_id'].values)

        if count:
            avg_experience /= count
            avg_num_review /= count

        features['num_of_reviewers'] = count
        features['num_of_bot_reviewers'] = bot
        features['avg_reviewer_experience'] = avg_experience
        features['avg_reviewer_review_count'] = avg_num_review

        # 所有评审人涉及的change
        total_ralated_changes = changes_until_current[changes_until_current['change_id'].isin(related_change_ids)]
        if total_ralated_changes.shape[0] > 0:
            features['review_avg_rounds'] = np.mean(total_ralated_changes['revision_num'].values)
            features['review_avg_duration'] = np.mean(total_ralated_changes['duration'].values)
        elif changes_until_current.shape[0] > 0:
            features['review_avg_rounds'] = np.mean(changes_until_current['revision_num'].values)
            features['review_avg_duration'] = np.mean(changes_until_current['duration'].values)
        else:
            features['review_avg_rounds'] = default_rounds
            features['review_avg_duration'] = default_duration

        return features

    @property
    def meta_features(self):
        features = {}

        label_list = self.change.labels
        score_sum = 0
        code_review_labels = [label for label in label_list if label.kind == 'Code-Review']
        for label in code_review_labels:
            score_sum += label.value
        score_count = len(code_review_labels)
        features['avg_score'] = 0 if score_count == 0 else score_sum / score_count

        features['commit_num'] = len(self.change.revisions)
        features['comment_num'] = self.change.comment_num
        current_comments_df = comment_list_df[comment_list_df['change_id'] == self.change.change_number]
        if current_comments_df.shape[0] == 0:
            features['comment_word_num'] = 0
            features['last_comment_mention'] = False
        else:
            comment_word_num = 0
            current_comments_df = current_comments_df.sort_values(by=['updated'])
            comment_messages = current_comments_df['message']
            for message in comment_messages:
                comment_word_num += len(message.split())
            features['comment_word_num'] = comment_word_num

            last_comment_message = comment_messages.iloc[-1]
            features['last_comment_mention'] = ('@' in last_comment_message)

        has_test = False
        files = self.change.files
        for file in files:
            file_name = file.name.lower()
            if 'test' in file_name:
                has_test = True
                break
        features['has_test_file'] = has_test

        features['description_readability'] = textstat.textstat.coleman_liau_index(self.change.subject)



        return features

    @property
    def file_features(self):
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

        # features['lines_added'] = lines_added
        # features['lines_deleted'] = lines_deleted

        features['num_of_directory'] = len(directories)
        features['subsystem_num'] = len(subsystems)

        features['files_added'] = files_added
        features['files_deleted'] = files_deleted
        features['files_modified'] = len(files) - files_deleted - files_added

        # Entropy is defined as: −Sum(k=1 to n)(pk∗log2pk). Note that n is number of files
        # modified in the change, and pk is calculated as the proportion of lines modified in file k among
        # lines modified in this code change.
        # modify_entropy = 0
        # if lines_changed:
        #     for file in files:
        #         lines_changed_in_file = file.lines_deleted + file.lines_inserted
        #         if lines_changed_in_file:
        #             pk = float(lines_changed_in_file) / lines_changed
        #             modify_entropy -= pk * np.log2(pk)
        #
        # features['modify_entropy'] = modify_entropy

        features['language_num'] = self.change.language_num
        features['file_type_num'] = self.change.file_type_num

        diff_result = self.diff_features
        features['lines_added'] = diff_result['lines_added']
        features['lines_deleted'] = diff_result['lines_deleted']
        features['lines_updated'] = diff_result['lines_updated']
        features['segs_added'] = diff_result['segs_added']
        features['segs_updated'] = diff_result['segs_updated']
        features['segs_deleted'] = diff_result['segs_deleted']

        features['modify_entropy'] = diff_result['modify_entropy']

        return features

    @property
    def description_features(self):
        subject = self.change.subject.lower()
        features = {'is_documentation': False, 'is_bug_fixing': False, 'is_feature': False,
                    'is_improve': False, 'is_refactor': False,
                    'description_length': len(subject.split())}

        for word in ['fix', 'bug', 'defect']:
            if word in subject:
                features['is_bug_fixing'] = True
                return features
        for word in ['doc', 'copyright', 'license']:
            if word in subject:
                features['is_documentation'] = True
                return features
        for word in ['improve']:
            if word in subject:
                features['is_improve'] = True
                return features
        for word in ['refactor']:
            if word in subject:
                features['is_refactor'] = True
        features['is_feature'] = True
        return features

    @property
    def diff_features(self):
        filepath = os.path.join(diff_root, f"{project}_{self.change.change_number}_diff.json")
        diff_json = json.load(open(filepath, 'r'))

        segs_added = segs_deleted = segs_updated = lines_added = lines_deleted = lines_updated = modify_entropy = 0
        total_line = 0

        try:
            files = list(diff_json.values())[0].values()
            for file in files:
                for content in file['content']:
                    change_type = list(content.keys())
                    if change_type == ['a']:
                        segs_deleted += 1
                        lines_deleted += len(content['a'])
                    elif change_type == ['a', 'b']:
                        segs_updated += 1
                        lines_updated += len(content['a'])
                    elif change_type == ['b']:
                        segs_added += 1
                        lines_added += len(content['b'])

            lines_changed = lines_added + lines_deleted + lines_updated
            if lines_changed:
                for file in files:
                    lines_changed_in_file = 0
                    for content in file['content']:
                        change_type = list(content.keys())
                        if change_type == ['a']:
                            lines_changed_in_file += len(content['a'])
                        elif change_type == ['a', 'b']:
                            lines_changed_in_file += len(content['a'])
                        elif change_type == ['b']:
                            lines_changed_in_file += len(content['b'])

                    if lines_changed_in_file:
                        pk = float(lines_changed_in_file) / lines_changed
                        modify_entropy -= pk * np.log2(pk)

        except IndexError:
            print('Error for {0}'.format(self.change.change_number))

        return {
            'segs_added': segs_added,
            'segs_updated': segs_updated,
            'segs_deleted': segs_deleted,
            'lines_added': lines_added,
            'lines_deleted': lines_deleted,
            'lines_updated': lines_updated,
            'modify_entropy': modify_entropy
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


if __name__ == '__main__':
    main()
