from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer

def get_preprocessor():
    onehot_cols = ['study_method', 'course', 'gender']
    # study_method could also be ordinal, but there aren't so much differences in median exam scores by category
    # course: there aren't big differences in median exam score
    # gender: no differences

    ordinal_cols = ['facility_rating', 'sleep_quality', 'exam_difficulty', 'internet_access']
    # facility_rating: low < medium < high
    # sleep_quality: poor < average < good
    # exam_difficulty: easy < moderate < hard, but median exam scores by category doesn't differ
    # internet_access: no < yes

    ordinal_ordering = [['low', 'medium', 'high'],
                        ['poor', 'average', 'good'],
                        ['easy', 'moderate', 'hard'],
                        ['no', 'yes']]

    num_cols = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

    standard_scaler = StandardScaler()
    onehot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ordinal_encoder = OrdinalEncoder(categories=ordinal_ordering)

    preprocessor = ColumnTransformer(transformers=[
        ('num', standard_scaler, num_cols),
        ('cat_onehot', onehot_encoder, onehot_cols),
        ('cat_ordinal', ordinal_encoder, ordinal_cols)
    ])

    return preprocessor