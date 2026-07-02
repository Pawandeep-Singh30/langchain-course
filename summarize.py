
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
load_dotenv()
def main():
    print("Hello from LangChain!")
    information = """Cristiano Ronaldo dos Santos Aveiro[a] (born 5 February 1985), nicknamed CR7, is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al-Nassr and the Portugal national team. Widely regarded as one of the greatest players in history and the greatest Portuguese player ever, he has won numerous individual accolades throughout his career, including five Ballon d'Or awards, a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes. He was also named the world's best player five times by FIFA.[note 3]
Ronaldo is one of the most decorated players in the history of professional football, having won 35 trophies in his career, including five UEFA Champions Leagues, two UEFA Nations Leagues and the UEFA European Championship. He holds the records for most goals (140) and assists (42) in the Champions League, goals (14) and assists (8) in the European Championship, and most international appearances (231), men's international goals (145) and most international victories (140). He is the only player to have scored 100 goals with four different clubs and to finish as top scorer in four different domestic leagues. He has made over 1,300 professional career appearances, the most by an outfield player, and has scored over 970 official senior career goals for club and country, making him the top goalscorer of all time.
Born in Funchal, Madeira, Ronaldo began his career with Sporting CP before signing with Manchester United in 2003. He gradually established himself as an integral player for the club, and won three consecutive Premier League titles, the Champions League, and the FIFA Club World Cup. For his performances in 2007–08, Ronaldo was awarded his first Ballon d'Or at age 23. In 2009, Ronaldo became the subject of the then-most expensive transfer in history when he joined Real Madrid in a deal worth €94 million (£80 million). At Madrid, he was at the forefront of the club's resurgence as a dominant European force, helping them win four Champions Leagues between 2014 and 2018, including the long-awaited La Décima in 2014, where he set the record for most goals scored in a Champions League season. He also won two La Liga titles, and became the club's all-time top goalscorer. At Madrid, he won the Ballon d'Or in 2013, 2014, 2016 and 2017.
Following issues with the club hierarchy, Ronaldo signed for Juventus in 2018 in a transfer worth a league record of €100 million, where he was pivotal in winning two consecutive Serie A titles, the Coppa Italia, and the Capocannoniere in 2021 as the league's top scorer. In 2021, he returned to Manchester United, but had his contract terminated in 2022 after a public dispute with the club's management. Ronaldo joined Al-Nassr in 2023, and led them to the Saudi Pro League title in 2026, while also finishing as the league top scorer back-to-back in 2024 and 2025.
Ronaldo made his international debut for Portugal in 2003 at the age of 18 and has earned more than 200 caps, making him history's most-capped male player.[7] He has played in twelve major tournaments. He scored his first international goal in Euro 2004, where he helped Portugal reach the final and subsequently made the Team of the Tournament. He assumed the captaincy of the national team ahead of Euro 2008, was named in the team of the tournament at Euro 2012, and led Portugal to their first major tournament title at Euro 2016. He scored a hattrick and was named in the Dream Team at the 2018 FIFA World Cup. At Euro 2020, he was named in the team of the tournament for the third time and received the Golden Boot as top scorer. He won the UEFA Nations Leagues in 2019 and 2025, finishing as the top scorer in both the tournaments. In 2026, he became the first player to score in six World Cup tournaments.
One of the world's most marketable and famous athletes, Ronaldo has been ranked by Sportico as the third highest-paid athlete of all time in April 2026. In 2026, Ronaldo appeared on the Forbes World’s Billionaires list for the first time at $1.2B net worth.[8] Time included him on their list of the 100 most influential people in the world in 2014. He is the most popular sportsperson on social media: he counts over 1 billion total followers across Facebook, Twitter, YouTube and Instagram, making him the first person to achieve that feat. Ronaldo was named in the UEFA Ultimate Team of the Year in 2015, the All-time UEFA Euro XI in 2016, and the Ballon d'Or Dream Team in 2020. In recognition of his record-breaking goalscoring success, he received special awards for Outstanding Career Achievement by FIFA in 2021.
    """
    summary_template = """
    Given the following information {information} about a person, i want u to create:
    1. a short summary
    2. achievements
    3. two interesting facts about them
    """
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    llm = ChatOllama(temperature=0, model="llama3.2")
    chain = summary_prompt_template | llm
    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()
