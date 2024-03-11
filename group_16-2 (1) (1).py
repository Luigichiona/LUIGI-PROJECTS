# Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBRegressor
from tqdm import tqdm
from scipy.stats import randint, uniform

publisher_dict = {'Association for Computational Linguistics;Association for Computational Linguistics': np.nan, 'European Language Resources Association (ELRA);European Language Resources Association (ELRA)': np.nan, 'European Language Resources Association;European Language Resources Association (ELRA)': np.nan, 'ATALA;ATALA': np.nan, 'MIT Press;MIT Press': np.nan, 'International Committee on Computational Linguistics;International Committee on Computational Linguistics': np.nan, 'The Association for Computational Linguistics and Chinese Language Processing (ACLCLP);The Association for Computational Linguistics and Chinese Language Processing (ACLCLP)': np.nan, 'European Association for Machine Translation;European Association for Machine Translation': np.nan, 'Asian Federation of Natural Language Processing;Asian Federation of Natural Language Processing': np.nan, 'The COLING 2016 Organizing Committee;The COLING 2016 Organizing Committee': np.nan, 'Association for Machine Translation in the Americas;Association for Machine Translation in the Americas': np.nan, 'INCOMA Ltd.;INCOMA Ltd.': np.nan, 'Aslib;Aslib': np.nan, 'The COLING 2012 Organizing Committee;The COLING 2012 Organizing Committee': np.nan, 'Coling 2010 Organizing Committee;Coling 2010 Organizing Committee': np.nan, 'COLING;COLING': np.nan, 'Coling 2008 Organizing Committee;Coling 2008 Organizing Committee': np.nan, 'ATALA (Association pour le Traitement Automatique des Langues);ATALA': np.nan, 'International Committee for Computational Linguistics;International Committee on Computational Linguistics': np.nan, 'NLP Association of India;NLP Association of India': np.nan, 'INCOMA Ltd. Shoumen, BULGARIA;INCOMA Ltd.': np.nan, 'Dublin City University and Association for Computational Linguistics;Dublin City University and Association for Computational Linguistics': np.nan, 'NLP Association of India (NLPAI);NLP Association of India': np.nan, 'ATALA/AFCP;ATALA': np.nan, 'Association for Computational Linguistics and Dublin City University;Dublin City University and Association for Computational Linguistics': np.nan, 'Chinese Information Processing Society of China;Chinese Information Processing Society of China': np.nan, 'Global Wordnet Association;Global Wordnet Association': np.nan, 'Springer;Springer': np.nan, 'AFCP - ATALA;AFCP - ATALA': np.nan, 'City University of Hong Kong;City University of Hong Kong': np.nan, 'Institute of Digital Enhancement of Cognitive Processing, Waseda University;Institute of Digital Enhancement of Cognitive Processing, Waseda University': np.nan, 'ATALA et AFCP;AFCP - ATALA': np.nan, 'Link√∂ping University Electronic Press, Sweden;Link√∂ping University Electronic Press, Sweden': np.nan, 'Association pour le Traitement Automatique des Langues;ATALA': np.nan, 'Link√∂ping University Electronic Press;Link√∂ping University Electronic Press, Sweden': np.nan, 'Association for Computational Lingustics;Association for Computational Linguistics': np.nan, 'Australasian Language Technology Association;Australasian Language Technology Association': np.nan, 'International Conference on Computational Linguistics;International Conference on Computational Linguistics': np.nan, 'University of Tartu Library;University of Tartu Library': np.nan, 'Northern European Association for Language Technology (NEALT);Northern European Association for Language Technology (NEALT)': np.nan, 'LiU Electronic Press;Link√∂ping University Electronic Press, Sweden': np.nan, 'Department of Linguistics, Chulalongkorn University;Department of Linguistics, Chulalongkorn University': np.nan, 'Tsinghua University Press;Tsinghua University Press': np.nan, 'NOVA CLUNL, Portugal;NOVA CLUNL, Portugal': np.nan, 'AsLing;AsLing': np.nan, 'Faculty of Computer Science, Universitas Indonesia;Faculty of Computer Science, Universitas Indonesia': np.nan, 'University of Tartu, Estonia;University of Tartu Library': np.nan, 'University of Tartu Press;University of Tartu Library': np.nan, 'The Korean Society for Language and Information (KSLI);The Korean Society for Language and Information (KSLI)': np.nan, 'The National University (Phillippines);The National University (Phillippines)': np.nan, 'European Language Resources association;European Language Resources Association (ELRA)': np.nan, 'De La Salle University, Manila, Philippines;De La Salle University, Manila, Philippines': np.nan, 'Department of English, National Chengchi University;Department of English, National Chengchi University': np.nan, 'Sociedade Brasileira de Computa√ß√£o;Sociedade Brasileira de Computa√ß√£o': np.nan, 'Kyung Hee University;Kyung Hee University': np.nan, 'Carnegy Mellon University;Carnegy Mellon University': np.nan, 'The Association for Machine Translation in the Americas;Association for Machine Translation in the Americas': np.nan, 'COLIPS PUBLICATIONS;COLIPS PUBLICATIONS': np.nan, 'PACLIC 14 Organizing Committee;PACLIC 14 Organizing Committee': np.nan, 'CSLI Publications;CSLI Publications': np.nan, 'National Cheng Kung University, Taiwan, R.O.C.;National Cheng Kung University, Taiwan, R.O.C.': np.nan, 'International Committee on Computational Linguistics (ICCL);International Committee on Computational Linguistics': np.nan, 'The Korean Society for Language and Information;The Korean Society for Language and Information (KSLI)': np.nan, '-;-': np.nan, 'Department of Computational Linguistics, IBL ‚Äì BAS;Department of Computational Linguistics, IBL ‚Äì BAS': np.nan, 'Institute for Research in Cognitive Science;Institute for Research in Cognitive Science': np.nan, 'Chinese and Oriental Languages Information Processing Society;Chinese and Oriental Languages Information Processing Society': np.nan, 'International Workshop on Spoken Language Translation;International Workshop on Spoken Language Translation': np.nan, 'Charles University in Prague, Matfyzpress, Prague, Czech Republic;Charles University in Prague, Matfyzpress, Prague, Czech Republic': np.nan, 'Association for Computational Linguistics and Chinese Language Processing;The Association for Computational Linguistics and Chinese Language Processing (ACLCLP)': np.nan, 'Uppsala University, Uppsala, Sweden;Uppsala University, Uppsala, Sweden': np.nan, 'Logico-Linguistic Society of Japan;Logico-Linguistic Society of Japan': np.nan, 'College Publications;College Publications': np.nan, 'Simon Fraser University;Simon Fraser University': np.nan, 'DFKI GmbH;DFKI GmbH': np.nan, 'Institute of Linguistics, Academia Sinica;Institute of Linguistics, Academia Sinica': np.nan, 'Special Interest Group on Discourse and Dialogue (SIGdial);Special Interest Group on Discourse and Dialogue (SIGdial)': np.nan, 'EDB-tjensten for humanistiske fag, University of Trondheim, Norway;EDB-tjensten for humanistiske fag, University of Trondheim, Norway': np.nan, 'University of Joensuu, Finland;University of Joensuu, Finland': np.nan, 'Association for Computational Linguistics and The Asian Federation of Natural Language Processing;Association for Computational Linguistics and The Asian Federation of Natural Language Processing': np.nan, 'Department of Linguistics, Uppsala University, Sweden;Department of Linguistics, Uppsala University, Sweden': np.nan, 'Institute of Lexicography, Institute of Linguistics, University of Iceland, Iceland;Institute of Lexicography, Institute of Linguistics, University of Iceland, Iceland': np.nan, 'KONVENS 2021 Organizers;KONVENS 2021 Organizers': np.nan, 'Workshop on Asian Translation;Workshop on Asian Translation': np.nan, 'Department of Linguistics, Computational Linguistics, Stockholm University, Sweden;Department of Linguistics, Computational Linguistics, Stockholm University, Sweden': np.nan, 'Linguistic Department, Yale University;Linguistic Department, Yale University': np.nan, 'KONVENS 2022 Organizers;KONVENS 2022 Organizers': np.nan, '√öFAL MFF UK;√öFAL MFF UK': np.nan, 'International Conference on Spoken Language Translation;International Conference on Spoken Language Translation': np.nan, 'Baltic Journal of Modern Computing;Baltic Journal of Modern Computing': np.nan, 'Department of Linguistics, Norwegian University of Science and Technology, Norway;Department of Linguistics, Norwegian University of Science and Technology, Norway': np.nan, 'Institut for Datalingvistik, Handelsh√∏jskolen i K√∏benhavn, Denmark;Institut for Datalingvistik, Handelsh√∏jskolen i K√∏benhavn, Denmark': np.nan, '?? Not mentionned on TOC;-': np.nan, 'Centrum f√∂r datorlingvistik, Uppsala University, Sweden;Centrum f√∂r datorlingvistik, Uppsala University, Sweden': np.nan, 'Department of General Linguistics, University of Helsinki, Finland;Department of General Linguistics, University of Helsinki, Finland': np.nan, 'Assocation for Computational Linguistics;Association for Computational Linguistics': np.nan, 'Center for Sprogteknologi, University of Copenhagen, Denmark;Center for Sprogteknologi, University of Copenhagen, Denmark': np.nan, 'European Language Resource Association;European Language Resources Association (ELRA)': np.nan, 'Aslib Proceedings;Aslib': np.nan, 'University of Antwerp;University of Antwerp': np.nan, 'INCOMA Inc.;INCOMA Ltd.': np.nan, 'Incoma Ltd., Shoumen, Bulgaria;INCOMA Ltd.': np.nan, 'Dublin City University and the Association for Computational Linguistics;Dublin City University and Association for Computational Linguistics': np.nan, 'Special Interest Group on Controlled Natural Language;Special Interest Group on Controlled Natural Language': np.nan, 'Spr√•kdata, University of Gothenburg, Sweden;Spr√•kdata, University of Gothenburg, Sweden': np.nan, 'Charles University, Faculty of Mathematics and Physics, Institute of Formal and Applied Linguistics;': np.nan, 'Institut for Anvendt og Matematisk Lingvistik, University of Copenhagen, Denmark;Institut for Anvendt og Matematisk Lingvistik, University of Copenhagen, Denmark': np.nan, 'Springer Berlin Heidelberg;Springer': np.nan, 'Norwegian Computing Centre for the Humanities, Norway;Norwegian Computing Centre for the Humanities, Norway': np.nan, 'European Language Ressources Association;European Language Resources Association (ELRA)': np.nan, 'Internationales Begegnungs- und Forschungszentrum f√ºr Informatik (IBFI);Internationales Begegnungs- und Forschungszentrum f√ºr Informatik (IBFI)': np.nan, 'Mouton de Gruyter;Mouton de Gruyter': np.nan, 'Pergamon Press;Pergamon Press': np.nan, 'Northern European Association of Language Technology;Northern European Association for Language Technology (NEALT)': np.nan, 'Dublin City University;Dublin City University and Association for Computational Linguistics': np.nan, 'Association for Computational Linguistics, Shoumen, Bulgaria;Association for Computational Linguistics, Shoumen, Bulgaria': np.nan, 'The European Language Resources Association (ELRA);European Language Resources Association (ELRA)': np.nan, 'The Association for Computational Linguistics;Association for Computational Linguistics': np.nan, 'Northern European Association for Language Technology;Northern European Association for Language Technology (NEALT)': np.nan, 'Link√∂ping Electronic Press;Link√∂ping University Electronic Press, Sweden': np.nan}

# Read in the datasets
df = pd.read_json('train.json')
test_df = pd.read_json('test.json')

# Add a column 'testdataset' to each DataFrame with all 0s and 1s respectively
df['testdataset'] = 0
test_df['testdataset'] = 1

# create the author year column 
#  create an empty dict
authors_data = {}

# Iterate through the authors and add them to the dict # Own work
for authors, year in tqdm(zip(df['author'], df['year'])):
    if authors is None:
        continue
    for au in authors:
        if au not in authors_data:
            authors_data[au] = [year]
        else:
            authors_data[au].append(year)

# Calculate the average year for authors in the test dataset
predicted_years_test = []
predicted_years_df = []

# Iterate through the authors again and calculate the mean for all authors
for authors in tqdm(test_df['author']):
    if authors is None:
        predicted_years_test.append(np.nan)  # Append average for missing authors
        continue

    years = []
    for au in authors:
        if au in authors_data:
            years.extend(authors_data[au])

    if years:
        predicted_years_test.append(np.mean(years))
    else:
        predicted_years_test.append(np.nan)  # Append NaN if no author found


for authors in tqdm(df['author']):
    if authors is None:
        predicted_years_df.append(np.nan)  # Append average for missing authors
        continue

    years = []
    for au in authors:
        if au in authors_data:
            years.extend(authors_data[au])

    if years:
        predicted_years_df.append(np.mean(years))
    else:
        predicted_years_df.append(np.nan)  # Append NaN if no author found

# Add the prediction for those authors to the DataFrame
test_df['average_author_year'] = predicted_years_test
df['average_author_year'] = predicted_years_df

# Concatenate the two DataFrames
combined_df = pd.concat([df, test_df], ignore_index=True)

# Feature engineering 
# Define a pattern to search for # Reference 1
pattern = r'\b(19[0-9]{2}|20[0-1][0-9]|202[0-3])\b'

# Extract the year from titles, publishers, and abstracts # Own Code
combined_df['title_year'] = combined_df['title'].str.extract(pattern, expand=False) 
combined_df['publisher_year'] = combined_df['publisher'].str.extract(pattern, expand=False) 
combined_df['abstract_year'] = combined_df['abstract'].str.extract(pattern, expand=False) 

# Convert year columns to numeric type # Own Code 
combined_df['title_year'] = pd.to_numeric(combined_df['title_year'], errors='coerce')
combined_df['publisher_year'] = pd.to_numeric(combined_df['publisher_year'], errors='coerce')
combined_df['abstract_year'] = pd.to_numeric(combined_df['abstract_year'], errors='coerce')

# Count the amount of authors and length of the: abstract, title, publisher and editor # Own Code 
combined_df['authorcount'] = combined_df['author'].apply(lambda x: len(x) if isinstance(x, list) else 0)
combined_df['abstract_length'] = combined_df['abstract'].apply(lambda x: len(x) if isinstance(x, str) else 0)
combined_df['title_length'] = combined_df['title'].apply(lambda x: len(x) if isinstance(x, str) else 0)
combined_df['editor_length'] = combined_df['editor'].apply(lambda x: len(x) if isinstance(x, str) else 0)
combined_df['publisher_length'] = combined_df['publisher'].apply(lambda x: len(x) if isinstance(x, str) else 0)

# Compute the average for each column # Own Code 
average_year_abstract = combined_df['abstract_year'].mean()
average_year_title = combined_df['title_year'].mean()
average_year_publisher = combined_df['publisher_year'].mean()

combined_df['title_year'] = combined_df[['title_year']].fillna({
    'title_year': average_year_title}).bfill(axis=1).iloc[:, 0]

combined_df['abstract_year'] = combined_df[['abstract_year']].fillna({
    'abstract_year': average_year_abstract}).bfill(axis=1).iloc[:, 0]

combined_df['publisher_year'] = combined_df[['publisher_year']].fillna({
    'publisher_year': average_year_publisher}).bfill(axis=1).iloc[:, 0]

combined_df['combined_year'] = combined_df[['title_year', 'publisher_year', 'abstract_year']].fillna({
    'abstract_year': average_year_abstract
}).bfill(axis=1).iloc[:, 0]

# Impute missing values for authors and editors # Own Code 
df['publisher'] = df.groupby('year')['publisher'].transform(lambda x: x.fillna(x.mode().iloc[0] if not x.mode().empty else 'Unknown Publisher'))
df['editor'] = df.groupby('year')['editor'].transform(lambda x: x.fillna('No ditor'))

# Use the map function to replace the values in the 'publisher' column # Reference 2
combined_df['publisher'] = combined_df['publisher'].map(publisher_dict).fillna(combined_df['publisher'])

#Fill the empty publishers # Own code
combined_df['publisher'] = combined_df['publisher'].fillna('Unknown Publisher')
combined_df['abstract'] = combined_df['abstract'].fillna('Unknown Abstract')

# Convert lists to strings for title, publisher, and abstract # Own code
combined_df['title'] = combined_df['title'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
combined_df['publisher'] = combined_df['publisher'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
combined_df['abstract'] = combined_df['abstract'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)

# Split the combined dataset based on the 'testdataset' column # Own code
df = combined_df[combined_df['testdataset'] == 0].drop(columns='testdataset')
test_df = combined_df[combined_df['testdataset'] == 1].drop(columns='testdataset')

# Pre processing 
# Define the features # Own code
numeric_features = ['title_year', 'publisher_year', 'abstract_year' ,
                      'authorcount', 'abstract_length','title_length' , 'editor_length', 'combined_year'
                    , 'average_author_year' 
                    ]
categorical_features = ['ENTRYTYPE', 'publisher']

# Define the transformation for each of the type of feature # Own code
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(min_frequency=2, handle_unknown='infrequent_if_exist')

# Define the vectorizer and max features and stop words # Baseline
tfidf_vectorizer_title = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_vectorizer_abstract = TfidfVectorizer(max_features=5000, stop_words='english')
tfidf_vectorizer_publisher = TfidfVectorizer(max_features=1000, stop_words='english')

# Define the pre processor steps # Baseline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('tfidf_title', tfidf_vectorizer_title, 'title'),
        ('tfidf_abstract', tfidf_vectorizer_abstract, 'abstract'),
        ('tfidf_publisher', tfidf_vectorizer_publisher, 'publisher')
    ])

# Define the XGB model # Own code 
xgb_model = XGBRegressor(n_estimators=440, 
                         random_state=42, 
                         learning_rate=0.347, 
                         max_depth=8, 
                         nthread=-1)

# Define the pipeline steps # Own code 
xgb_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('xgb_model', xgb_model)])

# Define features and target variable # Own code 
features = ['ENTRYTYPE', 'title', 'editor', 'publisher', 'abstract', 'author', 'authorcount', 'abstract_length', 'title_length', 'combined_year', 'editor_length',
            'publisher_length', 'title_year', 'publisher_year', 'abstract_year', 'average_author_year']
target = 'year'


# Train the model (XGB) on full data  # Own code 
xgb_trained = xgb_pipeline.fit(df[features], df[target])

# train test split and param search # Own code 
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=41)

# Train the model (XGB) on split data # Own code 
xgb_trained = xgb_pipeline.fit(X_train, y_train)

# Own code 
param_dist = {
    'xgb_model__n_estimators': randint(250, 550),
    'xgb_model__learning_rate': (0.2, 0.5), 
    'xgb_model__max_depth': randint(6, 9)  
}

# Own code 
# Initialize randomizedsearch
random_search = RandomizedSearchCV( 
    xgb_pipeline, param_dist, n_iter=20, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)

random_search.fit(df[features], df[target])
best_random_model = random_search.best_estimator_

# Extract the best params
print("Best Hyperparameters (Random Search):")
print(random_search.best_params_)

# Prediction on test set
# Make predictions
xgb_predictions = xgb_trained.predict(X_test)

# Evaluate the models
xgbmae2 = mean_absolute_error(y_test, xgb_predictions)
print(f'Mean Absolute Error XGB: {xgbmae2}')

rounded_pred = (np.round(xgb_predictions).astype(int))

# Evaluate the models
xgbmae2 = mean_absolute_error(y_test, rounded_pred)
print(f'Mean Absolute Error XGB (Rounded): {xgbmae2}')

# Create a prediction file # Own code 
pred = xgb_trained.predict(test_df)
rounded_pred = (np.round(pred).astype(int))
test_df['year'] = pred

pred_df = pd.DataFrame({'year': rounded_pred},)

# Save DataFrame to a JSON file # Own code 
pred_df.to_json("predicted.json", orient='records', indent=2)