projects = ['Libreoffice', 'Eclipse', 'Gerrithub']
project = projects[1]
data_folder = "/Users/aobang/Documents/学习资料/毕业设计/数据集/my_data/Data"
# data_folder = "../../Data"
root = f"{data_folder}/{project}"
change_folder = "change"
change_directory_path = f'{root}/{change_folder}'
changes_root = f"{root}/changes"
diff_root = f'{root}/diff'

comment_root = f'{root}/comment'

result_folder = "../../Results"
result_project_folder = f"{result_folder}/{project}"

target = 'status'
seed = 2021
folds = 11
runs = 10
account_list_filepath = f'{root}/{project}_account_list.csv'
change_list_filepath = f'{root}/{project}_change_list.csv'
selected_change_list_filepath = f'{root}/{project}_selected_change_list.csv'

comment_list_filepath = f'{root}/{project}_comment_list.csv'

features_group = {
    'author': ['author_experience', 'author_merge_ratio', 'author_changes_per_week',
               'author_merge_ratio_in_project', 'total_change_num', 'author_review_num',
               'is_reviewer',
               'author_subsystem_change_num', 'author_subsystem_change_merge_ratio', 'author_avg_rounds',
               'author_contribution_rate', 'author_merged_change_num', 'author_abandoned_changes_num',
               'author_avg_duration'],
    'text': ['description_length', 'is_documentation', 'is_bug_fixing', 'is_feature', 'is_improve', 'is_refactor'],
    'meta': ['commit_num', 'comment_num', 'comment_word_num', 'last_comment_mention', 'has_test_file',
             'description_readability', 'is_responded', 'first_response_duration'],
    'project': ['project_changes_per_week', 'project_merge_ratio', 'changes_per_author',
                'project_author_num', 'project_duration_per_merged_change', 'project_commits_per_merged_change',
                'project_comments_per_merged_change', 'project_file_num_per_merged_change',
                'project_churn_per_merged_change',
                'project_duration_per_abandoned_change', 'project_commits_per_abandoned_change',
                'project_comments_per_abandoned_change', 'project_file_num_per_abandoned_change',
                'project_churn_per_abandoned_change',
                'project_additions_per_week', 'project_deletions_per_week', 'workload'],
    'reviewer': ['num_of_reviewers', 'num_of_bot_reviewers', 'avg_reviewer_experience', 'avg_reviewer_review_count',
                 'review_avg_rounds', 'review_avg_duration'],
    'code': ['lines_added', 'lines_updated', 'lines_deleted', 'files_added', 'files_deleted', 'files_modified',
             'num_of_directory', 'modify_entropy', 'subsystem_num',
             'language_num', 'file_type_num', 'segs_added', 'segs_deleted', 'segs_updated', 'modified_code_ratio',
             'test_churn', 'src_churn'],
    'social': ['degree_centrality', 'closeness_centrality', 'betweenness_centrality',
               'eigenvector_centrality', 'clustering_coefficient', 'k_coreness'],
}

late_features = ["duration", "number_of_message", "number_of_revision",
                    "avg_delay_between_revision", "weighted_approval_score"]


def get_initial_feature_list() -> [str]:
    feature_list = []
    for group in features_group:
        feature_list.extend(features_group[group])
    return feature_list


initial_feature_list = get_initial_feature_list()