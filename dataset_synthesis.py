from tokenization import safe_first_token_id_variants, decode_if_bytes

from anthropic import Anthropic
from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
import json
from json.decoder import JSONDecodeError
from os.path import isfile
from tqdm import tqdm
from dataclasses import dataclass
from typing import Literal, Callable
from beartype import beartype

Checkpoint = str
Language = str

Messages = list[dict[Literal["user", "assistant", "system"], str]]

@beartype
def cached_completion( client: Anthropic,
                       messages: Messages,
                       model: str = "claude-3-5-sonnet-20240620",
                       discard_cached: bool = False,
                       cache_filename: str = "claude-completions.json" ) -> str:

    assert cache_filename.endswith(".json")
    
    if not isfile(cache_filename):
        with open(cache_filename, "w") as f:
            json.dump([], f)

    with open(cache_filename, "r") as f:
        cached: list[tuple[Messages, str]] = json.load(f)

    if not discard_cached:
        for cached_messages, cached_completion in cached:
            if messages == cached_messages:
                return cached_completion
    else:
        for i, (cached_messages, cached_completion) in enumerate(cached):
            if messages == cached_messages:
                cached = cached[:i] + cached[i+1:]
                break

    completion = client.messages.create(
        system = messages[0]["content"] if messages[0]["role"] == "system" else None,
        messages = messages[1:] if messages[0]["role"] == "system" else messages,
        model = model,
        max_tokens = 4096
    ).content[0].text

    cached.append((messages, completion))

    with open(cache_filename, "w") as f:
        json.dump(cached, f)

    return completion

TRANSLATE_EACH_TOKEN_SYSTEM_PROMPT = """You are presented with a tokenized text in some source langauge. Please translate the text the another target language. You should output a list of pairs. The first element of each pair should be the token in the source language. The second element should be:
* If the token in the source language is either a stop word or not a word at all, `null`.
* If the token in the source language is a whole word, or the first token of a word, the closest translation of the whole word to the target language, taking the context of the whole text into consideration.
* If the token in the source language belongs to a word, but is not the first token of this word, `null`.
Please output the list and no other text whatsoever. The output should be valid JSON."""

TRANSLATE_EACH_TOKEN_USER_PROMPT = """Please translate from {source_language} to {target_language}.

---=== TEXT ===---

{text}


---=== TOKENIZED TEXT ===---

{tokenized_text}"""

TRANSLATE_EACH_TOKEN_FEW_SHOT_EXAMPLES_OLD: Messages = [
    {
        "role": "user",
        "content": TRANSLATE_EACH_TOKEN_USER_PROMPT.format(
            source_language = "English",
            target_language = "French",
            text = """For today’s post, I’d like to take a look at California’s voter initiative to legalize pot. If the measure passes, and the sky doesn’t fall, many other states will probably be looking at similar law changes in the near future. Our drug policy of the last century has simply not worked, and it’s heartening to see a state attempting to legalize marijuana. The statistics on marijuana arrests are really shocking. According to the Drug Policy Alliance, which is in favor of legalization, blacks are arrested for marijuana possession between four and twelve times more than whites in California, even though studies have consistently shown that whites smoke more pot than blacks. In the last ten years, around 500,000 people have been arrested for possession. That’s absurd! Think about how expensive that is for the criminal justice system. California spends $216,000 for each juvenile inmate in its prison system, yet it spends only $8,000 per student in the Oakland school system. It seems to me that if you really want to limit drug use, it’d make more sense to spend more money keeping kids in school, helping them achieve. The economic benefits of legalizing marijuana are mind blowing. If marijuana was legalized and taxed at the same rate of tobacco, the money we would save on law enforcement and gain in tax revenue equals about $17 billion. As Nicholas Kristof notes, that is enough money to send every three and four year old in a poor neighborhood to pre-school. Or we could spend that money improving public school education. Or we could use the money to shore up border defense. Whatever we do, $17 billion is not exactly a trivial amount. For me, the biggest reason to legalize marijuana is to hurt the cartels. Immigration has emerged as a hot button issue recently, with Arizona passing a draconian immigration law and many similar propositions being considered by other states. People are worried about violence, and understandably so. No one wants to have foreign drug dealers operating in their back yard. But no matter how many laws we pass, or how much money we spend, marijuana from Mexico and other Latin American countries will always find a way across the border. Drug importers are smart, and the demand is so high that increased patrols by border agents and harsher prison sentences will not act as an effective deterrent. America will always have a demand for marijuana, and that means as long as the drug stays illegal, violent drug cartels will operate in our borders.""",
            tokenized_text = """['For', ' today', 'âĢ', 'Ļ', 's', ' post', ',', ' I', 'âĢ', 'Ļ', 'd', ' like', ' to', ' take', ' a', ' look', ' at', ' California', 'âĢ', 'Ļ', 's', ' voter', ' initiative', ' to', ' legalize', ' pot', '.', ' If', ' the', ' measure', ' passes', ',', ' and', ' the', ' sky', ' doesn', 'âĢ', 'Ļ', 't', ' fall', ',', ' many', ' other', ' states', ' will', ' probably', ' be', ' looking', ' at', ' similar', ' law', ' changes', ' in', ' the', ' near', ' future', '.', ' Our', ' drug', ' policy', ' of', ' the', ' last', ' century', ' has', ' simply', ' not', ' worked', ',', ' and', ' it', 'âĢ', 'Ļ', 's', ' heart', 'ening', ' to', ' see', ' a', ' state', ' attempting', ' to', ' legalize', ' marijuana', '.', ' The', ' statistics', ' on', ' marijuana', ' arrests', ' are', ' really', ' shocking', '.', ' According', ' to', ' the', ' Drug', ' Policy', ' Alliance', ',', ' which', ' is', ' in', ' favor', ' of', ' legalization', ',', ' blacks', ' are', ' arrested', ' for', ' marijuana', ' possession', ' between', ' four', ' and', ' twelve', ' times', ' more', ' than', ' whites', ' in', ' California', ',', ' even', ' though', ' studies', ' have', ' consistently', ' shown', ' that', ' whites', ' smoke', ' more', ' pot', ' than', ' blacks', '.', ' In', ' the', ' last', ' ten', ' years', ',', ' around', ' 500', ',', '000', ' people', ' have', ' been', ' arrested', ' for', ' possession', '.', ' That', 'âĢ', 'Ļ', 's', ' absurd', '!', ' Think', ' about', ' how', ' expensive', ' that', ' is', ' for', ' the', ' criminal', ' justice', ' system', '.', ' California', ' spends', ' $', '216', ',', '000', ' for', ' each', ' juvenile', ' inmate', ' in', ' its', ' prison', ' system', ',', ' yet', ' it', ' spends', ' only', ' $', '8', ',', '000', ' per', ' student', ' in', ' the', ' Oakland', ' school', ' system', '.', ' It', ' seems', ' to', ' me', ' that', ' if', ' you', ' really', ' want', ' to', ' limit', ' drug', ' use', ',', ' it', 'âĢ', 'Ļ', 'd', ' make', ' more', ' sense', ' to', ' spend', ' more', ' money', ' keeping', ' kids', ' in', ' school', ',', ' helping', ' them', ' achieve', '.', ' The', ' economic', ' benefits', ' of', ' legalizing', ' marijuana', ' are', ' mind', ' blowing', '.', ' If', ' marijuana', ' was', ' legalized', ' and', ' taxed', ' at', ' the', ' same', ' rate', ' of', ' tobacco', ',', ' the', ' money', ' we', ' would', ' save', ' on', ' law', ' enforcement', ' and', ' gain', ' in', ' tax', ' revenue', ' equals', ' about', ' $', '17', ' billion', '.', ' As', ' Nicholas', ' Krist', 'of', ' notes', ',', ' that', ' is', ' enough', ' money', ' to', ' send', ' every', ' three', ' and', ' four', ' year', ' old', ' in', ' a', ' poor', ' neighborhood', ' to', ' pre', '-', 'school', '.', ' Or', ' we', ' could', ' spend', ' that', ' money', ' improving', ' public', ' school', ' education', '.', ' Or', ' we', ' could', ' use', ' the', ' money', ' to', ' shore', ' up', ' border', ' defense', '.', ' Whatever', ' we', ' do', ',', ' $', '17', ' billion', ' is', ' not', ' exactly', ' a', ' trivial', ' amount', '.', ' For', ' me', ',', ' the', ' biggest', ' reason', ' to', ' legalize', ' marijuana', ' is', ' to', ' hurt', ' the', ' cartels', '.', ' Immigration', ' has', ' emerged', ' as', ' a', ' hot', ' button', ' issue', ' recently', ',', ' with', ' Arizona', ' passing', ' a', ' draconian', ' immigration', ' law', ' and', ' many', ' similar', ' propositions', ' being', ' considered', ' by', ' other', ' states', '.', ' People', ' are', ' worried', ' about', ' violence', ',', ' and', ' understandably', ' so', '.', ' No', ' one', ' wants', ' to', ' have', ' foreign', ' drug', ' dealers', ' operating', ' in', ' their', ' back', ' yard', '.', ' But', ' no', ' matter', ' how', ' many', ' laws', ' we', ' pass', ',', ' or', ' how', ' much', ' money', ' we', ' spend', ',', ' marijuana', ' from', ' Mexico', ' and', ' other', ' Latin', ' American', ' countries', ' will', ' always', ' find', ' a', ' way', ' across', ' the', ' border', '.', ' Drug', ' imp', 'orters', ' are', ' smart', ',', ' and', ' the', ' demand', ' is', ' so', ' high', ' that', ' increased', ' patrols', ' by', ' border', ' agents', ' and', ' harsher', ' prison', ' sentences', ' will', ' not', ' act', ' as', ' an', ' effective', ' deterrent', '.', ' America', ' will', ' always', ' have', ' a', ' demand', ' for', ' marijuana', ',', ' and', ' that', ' means', ' as', ' long', ' as', ' the', ' drug', ' stays', ' illegal', ',', ' violent', ' drug', ' cartels', ' will', ' operate', ' in', ' our', ' borders', '.']"""
        )
    },
    {
        "role": "assistant",
        "content": """[["For", "Pour"], [" today", " aujourd'hui"], ["â", null], ["Ģ", null], ["Ļ", null], ["s", null], [" post", " post"], [",", null], [" I", null], ["â", null], ["Ģ", null], ["Ļ", null], ["d", null], [" like", null], [" to", null], [" take", null], [" a", null], [" look", " regarder"], [" at", null], [" California", "Californie"], ["â", null], ["Ģ", null], ["Ļ", null], ["s", null], [" voter", " électeur"], [" initiative", " initiative"], [" to", null], [" legalize", " légaliser"], [" pot", "cannabis"], [".", null], [" If", " Si"], [" the", null], [" measure", " mesure"], [" passes", " passe"], [",", null], [" and", null], [" the", null], [" sky", " ciel"], [" doesn", null], ["â", null], ["Ģ", null], ["Ļ", null], ["t", null], [" fall", " tombe"], [",", null], [" many", " nombreux"], [" other", " autres"], [" states", " États"], [" will", null], [" probably", " probablement"], [" be", null], [" looking", " regarder"], [" at", null], [" similar", " similaires"], [" law", " loi"], [" changes", " changements"], [" in", null], [" the", null], [" near", " proche"], [" future", " avenir"], [".", null], [" Our", " Notre"], [" drug", " drogue"], [" policy", " politique"], [" of", null], [" the", null], [" last", " dernier"], [" century", " siècle"], [" has", null], [" simply", " simplement"], [" not", null], [" worked", " fonctionné"], [",", null], [" and", null], [" it", null], ["â", null], ["Ģ", null], ["Ļ", null], ["s", null], [" heart", " encourageant"], ["ening", null], [" to", null], [" see", " voir"], [" a", null], [" state", " État"], [" attempting", "tentant"], [" to", null], [" legalize", " légaliser"], [" marijuana", " marijuana"], [".", null], [" The", null], [" statistics", " statistiques"], [" on", null], [" marijuana", " marijuana"], [" arrests", " arrestations"], [" are", null], [" really", " vraiment"], [" shocking", " choquantes"], [".", null], [" According", " Selon"], [" to", null], [" the", null], [" Drug", " Drogue"], [" Policy", " Politique"], [" Alliance", " Alliance"], [",", null], [" which", null], [" is", null], [" in", null], [" favor", " faveur"], [" of", null], [" legalization", " légalisation"], [",", null], [" blacks", " noirs"], [" are", null], [" arrested", " arrêtés"], [" for", null], [" marijuana", " marijuana"], [" possession", " possession"], [" between", " entre"], [" four", " quatre"], [" and", null], [" twelve", " douze"], [" times", " fois"], [" more", " plus"], [" than", null], [" whites", " blancs"], [" in", null], [" California", " Californie"], [",", null], [" even", " même"], [" though", null], [" studies", " études"], [" have", null], [" consistently", " constamment"], [" shown", " montré"], [" that", null], [" whites", " blancs"], [" smoke", " fument"], [" more", " plus"], [" pot", " cannabis"], [" than", null], [" blacks", " noirs"], [".", null], [" In", " Dans"], [" the", null], [" last", " dernières"], [" ten", " dix"], [" years", " années"], [",", null], [" around", " environ"], [" 500", null], [",", null], ["000", null], [" people", " personnes"], [" have", null], [" been", null], [" arrested", " arrêtées"], [" for", null], [" possession", " possession"], [".", null], [" That", null], ["â", null], ["Ģ", null], ["Ļ", null], ["s", null], [" absurd", " absurde"], ["!", null], [" Think", " Pensez"], [" about", null], [" how", null], [" expensive", " coûteux"], [" that", null], [" is", null], [" for", null], [" the", null], [" criminal", " criminel"], [" justice", " justice"], [" system", " système"], [".", null], [" California", " Californie"], [" spends", " dépense"], [" $", null], ["216", null], [",", null], ["000", null], [" for", null], [" each", " chaque"], [" juvenile", " juvénile"], [" inmate", " détenu"], [" in", null], [" its", null], [" prison", " prison"], [" system", " système"], [",", null], [" yet", " pourtant"], [" it", null], [" spends", " dépense"], [" only", " seulement"], [" $", null], ["8", null], [",", null], ["000", null], [" per", " par"], [" student", " élève"], [" in", null], [" the", null], [" Oakland", " Oakland"], [" school", " scolaire"], [" system", " système"], [".", null], [" It", null], [" seems", " semble"], [" to", null], [" me", null], [" that", null], [" if", " si"], [" you", " vous"], [" really", " vraiment"], [" want", " voulez"], [" to", null], [" limit", " limiter"], [" drug", " drogue"], [" use", " utilisation"], [",", null], [" it", null], ["â", null], ["Ģ", null], ["Ļ", null], ["d", null], [" make", null], [" more", " plus"], [" sense", " sens"], [" to", null], [" spend", " dépenser"], [" more", " plus"], [" money", " argent"], [" keeping", " garder"], [" kids", " enfants"], [" in", null], [" school", " école"], [",", null], [" helping", " aidant"], [" them", " les"], [" achieve", " réussir"], [".", null], [" The", null], [" economic", " économiques"], [" benefits", " avantages"], [" of", null], [" legalizing", " légaliser"], [" marijuana", " marijuana"], [" are", null], [" mind", " esprit"], [" blowing", " soufflant"], [".", null], [" If", " Si"], [" marijuana", " marijuana"], [" was", null], [" legalized", " légalisée"], [" and", null], [" taxed", " taxée"], [" at", null], [" the", null], [" same", " même"], [" rate", " taux"], [" of", null], [" tobacco", " tabac"], [",", null], [" the", null], [" money", " argent"], [" we", " nous"], [" would", null], [" save", " économiserions"], [" on", null], [" law", " loi"], [" enforcement", " application"], [" and", null], [" gain", " gagner"], [" in", null], [" tax", " impôt"], [" revenue", " revenu"], [" equals", " équivaut"], [" about", " environ"], [" $", null], ["17", null], [" billion", " milliards"], [".", null], [" As", " Comme"], [" Nicholas", " Nicholas"], [" Krist", " Krist"], ["of", null], [" notes", " note"], [",", null], [" that", null], [" is", null], [" enough", " suffisamment"], [" money", " argent"], [" to", null], [" send", " envoyer"], [" every", " chaque"], [" three", " trois"], [" and", null], [" four", " quatre"], [" year", " ans"], [" old", null], [" in", null], [" a", null], [" poor", " pauvre"], [" neighborhood", " quartier"], [" to", null], [" pre", " école maternelle"], ["-", null], ["school", null], [".", null], [" Or", "Ou"], [" we", "nous"], [" could", " pourrions"], [" spend", " dépenser"], [" that", null], [" money", " argent"], [" improving", " améliorant"], [" public", " publique"], [" school", " école"], [" education", " éducation"], [".", null], [" Or", " Ou"], [" we", " nous"], [" could", " pourrions"], [" use", " utiliser"], [" the", null], [" money", " argent"], [" to", null], [" shore", " renforcer"], [" up", null], [" border", " frontière"], [" defense", " défense"], [".", null], [" Whatever", " Quoi que"], [" we", " nous"], [" do", " fassions"], [",", null], [" $", null], ["17", null], [" billion", " milliards"], [" is", null], [" not", null], [" exactly", " exactement"], [" a", null], [" trivial", " trivial"], [" amount", " montant"], [".", null], [" For", " Pour"], [" me", " moi"], [",", null], [" the", null], [" biggest", " plus grande"], [" reason", " raison"], [" to", null], [" legalize", " légaliser"], [" marijuana", " marijuana"], [" is", null], [" to", null], [" hurt", " nuire"], [" the", null], [" cartels", " cartels"], [".", null], [" Immigration", " Immigration"], [" has", null], [" emerged", " émergé"], [" as", null], [" a", null], [" hot", " brûlant"], [" button", " bouton"], [" issue", " problème"], [" recently", " récemment"], [",", null], [" with", null], [" Arizona", " Arizona"], [" passing", " adoptant"], [" a", null], [" draconian", " draconienne"], [" immigration", " immigration"], [" law", " loi"], [" and", null], [" many", " nombreuses"], [" similar", " similaires"], [" propositions", " propositions"], [" being", null], [" considered", " envisagées"], [" by", null], [" other", " autres"], [" states", " États"], [".", null], [" People", " Gens"], [" are", null], [" worried", " inquiets"], [" about", null], [" violence", " violence"], [",", null], [" and", null], [" understandably", " compréhensiblement"], [" so", null], [".", null], [" No", " Personne"], [" one", null], [" wants", " veut"], [" to", null], [" have", null], [" foreign", " étrangers"], [" drug", " drogue"], [" dealers", " trafiquants"], [" operating", " opérant"], [" in", null], [" their", null], [" back", " arrière"], [" yard", " cour"], [".", null], [" But", " Mais"], [" no", null], [" matter", " importe"], [" how", " combien"], [" many", " nombreuses"], [" laws", " lois"], [" we", " nous"], [" pass", " adoptons"], [",", null], [" or", null], [" how", " combien"], [" much", " beaucoup"], [" money", " argent"], [" we", " nous"], [" spend", " dépensons"], [",", null], [" marijuana", " marijuana"], [" from", null], [" Mexico", " Mexique"], [" and", null], [" other", " autres"], [" Latin", " Latino"], [" American", " Américains"], [" countries", " pays"], [" will", null], [" always", " toujours"], [" find", " trouver"], [" a", null], [" way", " moyen"], [" across", " à travers"], [" the", null], [" border", " frontière"], [".", null], [" Drug", " Drogue"], [" imp", " import"], ["orters", null], [" are", null], [" smart", " intelligents"], [",", null], [" and", null], [" the", null], [" demand", " demande"], [" is", null], [" so", " tellement"], [" high", " élevée"], [" that", null], [" increased", " augmentées"], [" patrols", " patrouilles"], [" by", null], [" border", " frontière"], [" agents", " agents"], [" and", null], [" harsher", " plus sévères"], [" prison", " prison"], [" sentences", " peines"], [" will", null], [" not", null], [" act", " agir"], [" as", null], [" an", null], [" effective", " efficace"], [" deterrent", " dissuasion"], [".", null], [" America", " Amérique"], [" will", null], [" always", " toujours"], [" have", null], [" a", null], [" demand", " demande"], [" for", null], [" marijuana", " marijuana"], [",", null], [" and", null], [" that", null], [" means", " signifie"], [" as", null], [" long", " longtemps"], [" as", null], [" the", null], [" drug", " drogue"], [" stays", " reste"], [" illegal", " illégale"], [",", null], [" violent", " violents"], [" drug", " drogue"], [" cartels", " cartels"], [" will", null], [" operate", " opéreront"], [" in", null], [" our", " nos"], [" borders", " frontières"], [".", null]]"""
    },
    {
        "role": "user",
        "content": TRANSLATE_EACH_TOKEN_USER_PROMPT.format(
            source_language = "English",
            target_language = "Russian",
            text = """Anarchists in solidarity with the purged immigrants of Agios Panteleimonas ventured once again to open the public playground which is kept locked by fascists in favor of segregation, leading to battle with riot police and five arrests. On Tuesday 9/06 anarchists in solidarity to immigrants who are being daily terrorised by fascist thugs of the Golden Dawn neonazi party and their local allies in the area of Agios Panteleimonas, moved to unblock the entrance of the local children playground which the fascists want to keep locked in an effort to impose segregation between greeks and immigrants, and "to preserve the blood purity of the white race"...While unblocking the playground the anarchists were attacked by fascists who were soon routed before the arrival of riot police forces who engaged the anarchists in battle with the aim of protecting the fascists. During the clashes one policeman was injured and five protesters were arrested on criminal charges. After the end of the clashes, a local greek father, Mr Tasoulas, defying the reign of terror in the area, took his son to play in the coveted playground. Soon they were surrounded by fascists who blocked the exit of the playground and threatened to linch the father calling him a traitor. After he managed to handle the child to a sympathetic neighbor, the fascists beat the father in full presence of the chief of the local police station. The strong police forces present at the scene then arrested the father and took him to the local police station, where his solicitor, a leading figure of the legal world and human rights activist, was piled with eggs by fascists who threatened her life. The new tension in the area comes after the euroelection ascent of LAOS, the fascist Popular Orthodox Alarm Party, to the 4th position with 7% of the vote.""",
            tokenized_text = """['An', 'arch', 'ists', ' in', ' solidarity', ' with', ' the', ' pur', 'ged', ' immigrants', ' of', ' Ag', 'ios', ' Pant', 'ele', 'imon', 'as', ' ventured', ' once', ' again', ' to', ' open', ' the', ' public', ' playground', ' which', ' is', ' kept', ' locked', ' by', ' fascists', ' in', ' favor', ' of', ' segregation', ',', ' leading', ' to', ' battle', ' with', ' riot', ' police', ' and', ' five', ' arrests', '.', ' On', ' Tuesday', ' 9', '/', '06', ' anarchists', ' in', ' solidarity', ' to', ' immigrants', ' who', ' are', ' being', ' daily', ' terror', 'ised', ' by', ' fascist', ' thugs', ' of', ' the', ' Golden', ' Dawn', ' neon', 'azi', ' party', ' and', ' their', ' local', ' allies', ' in', ' the', ' area', ' of', ' Ag', 'ios', ' Pant', 'ele', 'imon', 'as', ',', ' moved', ' to', ' un', 'block', ' the', ' entrance', ' of', ' the', ' local', ' children', ' playground', ' which', ' the', ' fascists', ' want', ' to', ' keep', ' locked', ' in', ' an', ' effort', ' to', ' impose', ' segregation', ' between', ' g', 'ree', 'ks', ' and', ' immigrants', ',', ' and', ' "', 'to', ' preserve', ' the', ' blood', ' purity', ' of', ' the', ' white', ' race', '"...', 'While', ' un', 'blocking', ' the', ' playground', ' the', ' anarchists', ' were', ' attacked', ' by', ' fascists', ' who', ' were', ' soon', ' routed', ' before', ' the', ' arrival', ' of', ' riot', ' police', ' forces', ' who', ' engaged', ' the', ' anarchists', ' in', ' battle', ' with', ' the', ' aim', ' of', ' protecting', ' the', ' fascists', '.', ' During', ' the', ' clashes', ' one', ' policeman', ' was', ' injured', ' and', ' five', ' protesters', ' were', ' arrested', ' on', ' criminal', ' charges', '.', ' After', ' the', ' end', ' of', ' the', ' clashes', ',', ' a', ' local', ' g', 'reek', ' father', ',', ' Mr', ' Tas', 'oul', 'as', ',', ' def', 'ying', ' the', ' reign', ' of', ' terror', ' in', ' the', ' area', ',', ' took', ' his', ' son', ' to', ' play', ' in', ' the', ' coveted', ' playground', '.', ' Soon', ' they', ' were', ' surrounded', ' by', ' fascists', ' who', ' blocked', ' the', ' exit', ' of', ' the', ' playground', ' and', ' threatened', ' to', ' l', 'inch', ' the', ' father', ' calling', ' him', ' a', ' traitor', '.', ' After', ' he', ' managed', ' to', ' handle', ' the', ' child', ' to', ' a', ' sympathetic', ' neighbor', ',', ' the', ' fascists', ' beat', ' the', ' father', ' in', ' full', ' presence', ' of', ' the', ' chief', ' of', ' the', ' local', ' police', ' station', '.', ' The', ' strong', ' police', ' forces', ' present', ' at', ' the', ' scene', ' then', ' arrested', ' the', ' father', ' and', ' took', ' him', ' to', ' the', ' local', ' police', ' station', ',', ' where', ' his', ' solicitor', ',', ' a', ' leading', ' figure', ' of', ' the', ' legal', ' world', ' and', ' human', ' rights', ' activist', ',', ' was', ' piled', ' with', ' eggs', ' by', ' fascists', ' who', ' threatened', ' her', ' life', '.', ' The', ' new', ' tension', ' in', ' the', ' area', ' comes', ' after', ' the', ' euro', 'election', ' ascent', ' of', ' LA', 'OS', ',', ' the', ' fascist', ' Popular', ' Orthodox', ' Al', 'arm', ' Party', ',', ' to', ' the', ' 4', 'th', ' position', ' with', ' 7', '%', ' of', ' the', ' vote', '.']"""
        )
    },
    {
        "role": "assistant",
        "content": """[["An", "Анар"], ["arch", null], ["ists", null], [" in", null], [" solidarity", " солидарность"], [" with", null], [" the", null], [" pur", " очищен"], ["ged", null], [" immigrants", " иммигранты"], [" of", null], [" Ag", " Агиос"], ["ios", null], [" Pant", " Пантелемионас"], ["ele", null], ["imon", null], ["as", null], [" ventured", " отважились"], [" once", null], [" again", " снова"], [" to", null], [" open", " открыть"], [" the", null], [" public", " общественную"], [" playground", " площадку"], [" which", null], [" is", null], [" kept", " держали"], [" locked", " запертой"], [" by", null], [" fascists", " фашисты"], [" in", null], [" favor", " пользу"], [" of", null], [" segregation", " сегрегации"], [",", null], [" leading", " привело"], [" to", null], [" battle", " битве"], [" with", null], [" riot", " бунтующей"], [" police", " полицией"], [" and", null], [" five", " пять"], [" arrests", " арестов"], [".", null], [" On", null], [" Tuesday", " вторник"], [" 9", null], ["/", null], ["06", null], [" anarchists", " анархисты"], [" in", null], [" solidarity", " солидарность"], [" to", null], [" immigrants", " иммигрантам"], [" who", null], [" are", null], [" being", null], [" daily", " ежедневно"], [" terror", " терроризированны"], ["ised", null], [" by", null], [" fascist", " фашистскими"], [" thugs", " головорезами"], [" of", null], [" the", null], [" Golden", " Золотой"], [" Dawn", " Зари"], [" neon", " неонацистской"], ["azi", null], [" party", " партии"], [" and", null], [" their", null], [" local", " местными"], [" allies", " союзниками"], [" in", null], [" the", null], [" area", " районе"], [" of", null], [" Ag", " Агиос"], ["ios", null], [" Pant", " Пантелемионас"], ["ele", null], ["imon", null], ["as", null], [",", null], [" moved", " переместились"], [" to", null], [" un", " разблокировать"], ["block", null], [" the", null], [" entrance", " вход"], [" of", null], [" the", null], [" local", " местной"], [" children", " детской"], [" playground", " площадки"], [" which", null], [" the", null], [" fascists", " фашисты"], [" want", " хотят"], [" to", null], [" keep", " держать"], [" locked", " запертой"], [" in", null], [" an", null], [" effort", " попытке"], [" to", null], [" impose", " навязать"], [" segregation", " сегрегацию"], [" between", " между"], [" g", " грек"], ["ree", null], ["ks", null], [" and", null], [" immigrants", " иммигрантами"], [",", null], [" and", null], [" \"", null], ["to", null], [" preserve", " сохранить"], [" the", null], [" blood", " кровную"], [" purity", " чистоту"], [" of", null], [" the", null], [" white", " белой"], [" race", " расы"], ["\"...", null], ["While", " Пока"], [" un", " разблокировали"], ["blocking", null], [" the", null], [" playground", " площадку"], [" the", null], [" anarchists", " анархисты"], [" were", null], [" attacked", " нападение"], [" by", null], [" fascists", " фашистов"], [" who", null], [" were", null], [" soon", " вскоре"], [" routed", " разгромлены"], [" before", " перед"], [" the", null], [" arrival", " прибытием"], [" of", null], [" riot", " отряда"], [" police", " полиции"], [" forces", null], [" who", null], [" engaged", " вступили"], [" the", null], [" anarchists", " анархистами"], [" in", null], [" battle", " битву"], [" with", null], [" the", null], [" aim", " целью"], [" of", null], [" protecting", " защитить"], [" the", null], [" fascists", " фашистов"], [".", null], [" During", " Во время"], [" the", null], [" clashes", " столкновений"], [" one", " один"], [" policeman", " полицейский"], [" was", null], [" injured", " ранен"], [" and", null], [" five", " пять"], [" protesters", " протестующих"], [" were", null], [" arrested", " арестованы"], [" on", null], [" criminal", " уголовным"], [" charges", " обвинениям"], [".", null], [" After", " После"], [" the", null], [" end", " окончания"], [" of", null], [" the", null], [" clashes", " столкновений"], [",", null], [" a", null], [" local", " местный"], [" g", " греческий"], ["reek", null], [" father", " отец"], [",", null], [" Mr", " Господин"], [" Tas", " Тасолас"], ["oul", null], ["as", null], [",", null], [" def", "бросая"], ["ying", null], [" the", null], [" reign", " царству"], [" of", null], [" terror", " террора"], [" in", null], [" the", null], [" area", " районе"], [",", null], [" took", " взял"], [" his", null], [" son", " сына"], [" to", null], [" play", " играть"], [" in", null], [" the", null], [" coveted", " желанной"], [" playground", " площадке"], [".", null], [" Soon", " Вскоре"], [" they", " они"], [" were", null], [" surrounded", " окружены"], [" by", null], [" fascists", " фашистами"], [" who", null], [" blocked", " заблокировали"], [" the", null], [" exit", " выход"], [" of", null], [" the", null], [" playground", " площадки"], [" and", null], [" threatened", " угрожали"], [" to", null], [" l", " линчевать"], ["inch", null], [" the", null], [" father", " отца"], [" calling", " называя"], [" him", " его"], [" a", null], [" traitor", " предателем"], [".", null], [" After", " После"], [" he", null], [" managed", " сумел"], [" to", null], [" handle", " передать"], [" the", null], [" child", " ребенка"], [" to", null], [" a", null], [" sympathetic", " сочувствующему"], [" neighbor", " соседу"], [",", null], [" the", null], [" fascists", " фашисты"], [" beat", " избили"], [" the", null], [" father", " отца"], [" in", null], [" full", " полном"], [" presence", " присутствии"], [" of", null], [" the", null], [" chief", " начальника"], [" of", null], [" the", null], [" local", " местного"], [" police", " полицейского"], [" station", " участка"], [".", null], [" The", null], [" strong", " большие"], [" police", " полицейские"], [" forces", " силы"], [" present", " присутствующие"], [" at", null], [" the", null], [" scene", " месте"], [" then", " затем"], [" arrested", " арестовали"], [" the", null], [" father", " отца"], [" and", null], [" took", " отвезли"], [" him", null], [" to", null], [" the", null], [" local", " местный"], [" police", " полицейский"], [" station", " участок"], [",", null], [" where", " где"], [" his", null], [" solicitor", " адвокат"], [",", null], [" a", null], [" leading", " ведущая"], [" figure", " фигура"], [" of", null], [" the", null], [" legal", " юридического"], [" world", " мира"], [" and", null], [" human", " человека"], [" rights", " Прав"], [" activist", " активист"], [",", null], [" was", null], [" piled", " забросана"], [" with", null], [" eggs", " яйцами"], [" by", null], [" fascists", " фашистами"], [" who", null], [" threatened", " угрожали"], [" her", null], [" life", " жизни"], [".", null], [" The", null], [" new", " новое"], [" tension", " напряжение"], [" in", null], [" the", null], [" area", " районе"], [" comes", " возникает"], [" after", " после"], [" the", null], [" euro", " евро"], ["election", null], [" ascent", " подъёма"], [" of", null], [" LA", " ЛАОС"], ["OS", null], [",", null], [" the", null], [" fascist", " фашистской"], [" Popular", " Народной"], [" Orthodox", " Православной"], [" Al", " Тревожности"], ["arm", null], [" Party", " Партии"], [",", null], [" to", null], [" the", null], [" 4", null], ["th", null], [" position", " позицию"], [" with", null], [" 7", null], ["%", null], [" of", null], [" the", null], [" vote", " голосов"], [".", null]]"""
    }
]

TRANSLATE_EACH_TOKEN_FEW_SHOT_EXAMPLES = [] # TO DO

@beartype
def translate_each_token(
        text: str,
        tokenizer: PreTrainedTokenizerBase,
        source_language: Language,
        target_language: Language,
        anthropic_client: Anthropic
    ) -> list[list[int] | None] | None:

    # note: implement prefix caching to reduce the API call cost by a lot
    
    tokenized_text: list[str] = tokenizer.tokenize(text)
    tokenized_text = [decode_if_bytes(token) for token in tokenized_text]
    # some tokenizers use Ġ instead of spaces, we don't want this
    # note that some tokenizers add other weird characters in other places, but those don't
    # seem common enough to matter
    tokenized_text = [token.replace("Ġ", " ") for token in tokenized_text]

    if source_language.strip().lower() == target_language.strip().lower():
        tokenized_text = [ safe_first_token_id_variants(token, tokenizer=tokenizer)
                           for token in tokenized_text ]
        return tokenized_text

    prompt = [
        { "role": "system",
          "content": TRANSLATE_EACH_TOKEN_SYSTEM_PROMPT },
        *TRANSLATE_EACH_TOKEN_FEW_SHOT_EXAMPLES,
        { "role": "user",
          "content": TRANSLATE_EACH_TOKEN_USER_PROMPT.format(
                source_language = source_language,
                target_language = target_language,
                text = text,
                tokenized_text = tokenized_text
            ) }
    ]

    token_translations = cached_completion( client = anthropic_client,
                                            messages = prompt )

    try:
        token_translations = json.loads(token_translations)
    except JSONDecodeError:
        return None

    valid_format =     isinstance(token_translations, list) \
                   and all(isinstance(t, list) for t in token_translations) \
                   and all(len(t) == 2 for t in token_translations) \
                   and all( isinstance(token, str) and isinstance(translation, str | None)
                            for token, translation in token_translations )
    if not valid_format:
        return None

    tokens_match =     len(tokenized_text) == len(token_translations) \
                   and all( token.strip(" ") == token_.strip(" ")
                            for token, (token_, translation) in
                                zip(tokenized_text, token_translations) )
    if not tokens_match:
        return None
           
    translations_with_adjusted_spaces: list[str | None] = []
    # this cannot be `for token, translation in token_translations` because the elements of
    # `tokenized_text`` and `translations` are not exactly the same, they are only the same up to
    # stripping spaces
    for token, (_, translation) in zip(tokenized_text, token_translations, strict=True):
        if translation is None:
            translations_with_adjusted_spaces.append(None)
            continue

        all_spaces = token.strip() == ""
        if all_spaces:
            if translation is not None:
                return None

            translations_with_adjusted_spaces.append(translation)
            continue

        n_leading_spaces = len(token) - len(token.lstrip(" "))
        n_trailing_spaces = len(token) - len(token.rstrip(" "))
        translations_with_adjusted_spaces.append(
            " " * n_leading_spaces + translation + " " * n_trailing_spaces
        )

    assert len(translations_with_adjusted_spaces) == len(tokenized_text)

    # WARNING: if some translations are more than one token long, we only keep the first token
    # and don't warn about it in any other way than this comment
    translations_with_adjusted_spaces = [
        (safe_first_token_id_variants(token, tokenizer=tokenizer) if token is not None else None)
        for token in translations_with_adjusted_spaces
    ]

    return translations_with_adjusted_spaces

@beartype
@dataclass
class TokenTranslations:
    text: str
    token_translations: list[dict[Language, list[int]] | None]

@beartype
def make_translated_token_dataset(
        texts: list[str],
        size: int,
        tokenizer: PreTrainedTokenizerBase,
        source_language: Language,
        target_languages: list[Language],
        anthropic_client: Anthropic,
        minimal_text_length: int | None = None
    ) -> list[TokenTranslations]:

    dataset = []
    
    with tqdm(desc="translating tokens", total=size) as progress_bar:
        for text in texts:
            if minimal_text_length is not None and len(text) < minimal_text_length:
                continue

            translations: dict[Language, list[list[int] | None] | None] = {
                target_language: translate_each_token( text = text,
                                                       tokenizer = tokenizer,
                                                       source_language = source_language,
                                                       target_language = target_language,
                                                       anthropic_client = anthropic_client )
                for target_language in target_languages
            }

            if not all(translation is not None for translation in translations.values()):
                continue

            n_tokens = len(translations[target_languages[0]])
            token_translations: list[dict[Language, list[int]] | None] = []
            for i in range(n_tokens):
                all_translations_exist = all( translation[i] is not None
                                              for translation in translations.values() )
                if not all_translations_exist:
                    token_translations.append(None)
                    continue

                token_translations.append({ language: translation[i]
                                            for language, translation in translations.items() })

            dataset.append(TokenTranslations(text=text, token_translations=token_translations))

            progress_bar.update(1)

            if len(dataset) >= size:
                return dataset

    assert False, f"Could not construct a token translation dataset of size {size} because too few texts were given and/or too many translations generated with the Claude API were discarded because Claude gave an invalid response."

@beartype
def load_opus_books_dataset(language: Language) -> list[str]:
    LANGUAGE_CODES = {
        "Catalan": "ca",
        "German": "de",
        "Greek": "el",
        "English": "en",
        "Esperanto": "eo",
        "Spanish": "es",
        "Finnish": "fi",
        "French": "fr",
        "Hungarian": "hu",
        "Italian": "it",
        "Dutch": "nl",
        "Norwegian": "no",
        "Polish": "pl",
        "Portuguese": "pt",
        "Russian": "ru",
        "Swedish": "sv"
    }

    assert language in LANGUAGE_CODES.keys(), f"Language is {language} but should be one of {list(LANGUAGE_CODES.keys())}."

    dataset = load_dataset("Helsinki-NLP/opus_books", f"de-{LANGUAGE_CODES[language]}")
    dataset = dataset["train"]["translation"]
    dataset: list[str] = [ datapoint[LANGUAGE_CODES[language]]
                           for datapoint in dataset ]
    
    return dataset

@beartype
def load_wmt_dataset(language: Language) -> list[str]:
    LANGUAGE_CODES = {
        "English": "en",
        "Chinese": "zh",
        "French": "fr",
        "Russian": "ru",
        "German": "de",
        "Czech": "cs",
        "Finnish": "fi",
        "Gujarati": "gu",
        "Kazakh": "kk",
        "Lithuanian": "lt",
    }

    assert language in LANGUAGE_CODES.keys(), f"Language is {language} but should be one of {list(LANGUAGE_CODES.keys())}."

    subset = f"{LANGUAGE_CODES[language]}-en" if language != "English" else "zh-en"
    dataset = load_dataset("wmt/wmt19", subset)
    dataset = dataset["validation"]["translation"]
    dataset: list[str] = [ datapoint[LANGUAGE_CODES[language]]
                           for datapoint in dataset ]
    
    return dataset

@beartype
def load_multiligual_dataset(dataset_name: str, language: Language) -> list[str]:
    load_dataset_fns: dict[str, Callable[[Language], str]] = {
        "opus_books": load_opus_books_dataset,
        "wmt": load_wmt_dataset
    }
    
    assert dataset_name in load_dataset_fns.keys(), f"dataset_name is '{dataset_name}' but should be one of {list(load_dataset_fns.keys())}"
    load_dataset_fn = load_dataset_fns[dataset_name]
    return load_dataset_fn(language=language)
