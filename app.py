import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import re
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import json
import os
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv
import traceback
import plotly.graph_objects as go
import schedule
import threading
import matplotlib.pyplot as plt

# 워드클라우드 추가
try:
    from wordcloud import WordCloud
except ImportError:
    st.error("wordcloud 패키지를 설치해주세요: pip install wordcloud")
    WordCloud = None

# 스케줄러 상태 클래스 추가
class SchedulerState:
    def __init__(self):
        self.is_running = False
        self.thread = None
        self.last_run = None
        self.next_run = None
        self.scheduled_jobs = []
        self.scheduled_results = []

# 전역 스케줄러 상태 객체 생성 (스레드 안에서 사용)
global_scheduler_state = SchedulerState()

# API 키 관리를 위한 세션 상태 초기화
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = None

# 환경 변수에서 API 키 로드 시도
load_dotenv()
if os.getenv('OPENAI_API_KEY'):
    st.session_state.openai_api_key = os.getenv('OPENAI_API_KEY')
elif 'OPENAI_API_KEY' in st.secrets:
    st.session_state.openai_api_key = st.secrets['OPENAI_API_KEY']

# 필요한 NLTK 데이터 다운로드
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# OpenAI API 키 설정 (실제 사용 시 환경 변수나 Streamlit secrets에서 가져오는 것이 좋습니다)
if 'OPENAI_API_KEY' in os.environ:
    openai.api_key = os.environ['OPENAI_API_KEY']
elif 'OPENAI_API_KEY' in st.secrets:
    openai.api_key = st.secrets['OPENAI_API_KEY']
elif os.getenv('OPENAI_API_KEY'):
    openai.api_key = os.getenv('OPENAI_API_KEY')

# 페이지 설정
st.set_page_config(page_title="뉴스 기사 도구", page_icon="📰", layout="wide")

# 사이드바 메뉴 설정
st.sidebar.title("뉴스 기사 도구")
menu = st.sidebar.radio(
    "메뉴 선택",
    ["뉴스 기사 크롤링", "기사 분석하기", "새 기사 생성하기", "뉴스 기사 예약하기"]
)

# 저장된 기사를 불러오는 함수
def load_saved_articles():
    if os.path.exists('saved_articles/articles.json'):
        with open('saved_articles/articles.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

# 기사를 저장하는 함수
def save_articles(articles):
    os.makedirs('saved_articles', exist_ok=True)
    with open('saved_articles/articles.json', 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

@st.cache_data
def crawl_naver_news(keyword, num_articles=5):
    """
    네이버 뉴스 기사를 수집하는 함수
    """
    url = f"https://search.naver.com/search.naver?where=news&query={keyword}"
    results = []
    
    try:
        # 페이지 요청
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 뉴스 아이템 찾기
        news_items = soup.select('div.sds-comps-base-layout.sds-comps-full-layout')
        
        # 각 뉴스 아이템에서 정보 추출
        for i, item in enumerate(news_items):
            if i >= num_articles:
                break
                
            try:
                # 제목과 링크 추출
                title_element = item.select_one('a.X0fMYp2dHd0TCUS2hjww span')
                if not title_element:
                    continue
                    
                title = title_element.text.strip()
                link_element = item.select_one('a.X0fMYp2dHd0TCUS2hjww')
                link = link_element['href'] if link_element else ""
                
                # 언론사 추출
                press_element = item.select_one('div.sds-comps-profile-info-title span.sds-comps-text-type-body2')
                source = press_element.text.strip() if press_element else "알 수 없음"
                
                # 날짜 추출
                date_element = item.select_one('span.r0VOr')
                date = date_element.text.strip() if date_element else "알 수 없음"
                
                # 미리보기 내용 추출
                desc_element = item.select_one('a.X0fMYp2dHd0TCUS2hjww.IaKmSOGPdofdPwPE6cyU > span')
                description = desc_element.text.strip() if desc_element else "내용 없음"
                
                results.append({
                    'title': title,
                    'link': link,
                    'description': description,
                    'source': source,
                    'date': date,
                    'content': ""  # 나중에 원문 내용을 저장할 필드
                })
                
            except Exception as e:
                st.error(f"기사 정보 추출 중 오류 발생: {str(e)}")
                continue
                
    except Exception as e:
        st.error(f"페이지 요청 중 오류 발생: {str(e)}")
        
    return results

# 기사 원문 가져오기
def get_article_content(url):
    try:
        response = requests.get(url, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 네이버 뉴스 본문 찾기
        content = soup.select_one('#dic_area')
        if content:
            text = content.text.strip()
            text = re.sub(r'\s+', ' ', text)  # 여러 공백 제거
            return text
            
        # 다른 뉴스 사이트 본문 찾기 (여러 사이트 대응 필요)
        content = soup.select_one('.article_body, .article-body, .article-content, .news-content-inner')
        if content:
            text = content.text.strip()
            text = re.sub(r'\s+', ' ', text)
            return text
            
        return "본문을 가져올 수 없습니다."
    except Exception as e:
        return f"오류 발생: {str(e)}"

# NLTK를 이용한 키워드 분석
def analyze_keywords(text, top_n=10):
    # 한국어 불용어 목록 (직접 정의해야 합니다)
    korean_stopwords = ['이', '그', '저', '것', '및', '등', '를', '을', '에', '에서', '의', '으로', '로']
    
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum() and len(word) > 1 and word not in korean_stopwords]
    
    word_count = Counter(tokens)
    top_keywords = word_count.most_common(top_n)
    
    return top_keywords

#워드 클라우드용 분석
def extract_keywords_for_wordcloud(text, top_n=50):
    if not text or len(text.strip()) < 10:
        return {}

    try:
        try:
            tokens = word_tokenize(text.lower())
        except Exception as e:
            st.warning(f"{str(e)} 오류발생")
            tokens = text.lower().split()
        
        stop_words = set()
        try:
            stop_words = set(stopwords.words('english'))
        except Exception:
            pass

        korea_stop_words = {
            '및', '등', '를', '이', '의', '가', '에', '는', '으로', '에서', '그', '또', '또는', '하는', '할', '하고',
                '있다', '이다', '위해', '것이다', '것은', '대한', '때문', '그리고', '하지만', '그러나', '그래서',
                '입니다', '합니다', '습니다', '요', '죠', '고', '과', '와', '도', '은', '수', '것', '들', '제', '저',
                '년', '월', '일', '시', '분', '초', '지난', '올해', '내년', '최근', '현재', '오늘', '내일', '어제',
                '오전', '오후', '부터', '까지', '에게', '께서', '이라고', '라고', '하며', '하면서', '따라', '통해',
                '관련', '한편', '특히', '가장', '매우', '더', '덜', '많이', '조금', '항상', '자주', '가끔', '거의',
                '전혀', '바로', '정말', '만약', '비롯한', '등을', '등이', '등의', '등과', '등도', '등에', '등에서',
                '기자', '뉴스', '사진', '연합뉴스', '뉴시스', '제공', '무단', '전재', '재배포', '금지', '앵커', '멘트',
                '일보', '데일리', '경제', '사회', '정치', '세계', '과학', '아이티', '닷컴', '씨넷', '블로터', '전자신문'
        }
        stop_words.update(korea_stop_words)

        # 1글자 이상이고 불용어가 아닌 토큰만 필터링
        filtered_tokens = [word for word in tokens if len(word) > 1 and word not in stop_words]
        
        # 단어 빈도 계산
        word_freq = {}
        for word in filtered_tokens:
            if word.isalnum():  # 알파벳과 숫자만 포함된 단어만 허용
                word_freq[word] = word_freq.get(word, 0) + 1
                
        # 빈도순으로 정렬하여 상위 n개 반환
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)

        if not sorted_words:
            return {"data": 1, "analysis": 1, "news": 1}
        
        return dict(sorted_words[:top_n])
    
    except Exception as e:
        st.error(f"오류발생 {str(e)}")
        return {"data": 1, "analysis": 1, "news": 1}
    

# 워드 클라우드 생성 함수

def generate_wordcloud(keywords_dict):
        if not WordCloud:
            st.warning("워드클라우드 설치안되어 있습니다.")
            return None
        try:
            wc= WordCloud(
                width=800,
                height=400,
                background_color = 'white',
                colormap = 'viridis',
                max_font_size=150,
                random_state=42
            ).generate_from_frequencies(keywords_dict)

            try:
                possible_font_paths=["NanumGothic.ttf", "이름"]

                font_path = None
                for path in possible_font_paths:
                    if os.path.exists(path):
                        font_path = path
                        break

                if font_path:
                    wc= WordCloud(
                        font_path=font_path,
                        width=800,
                        height=400,
                        background_color = 'white',
                        colormap = 'viridis',
                        max_font_size=150,
                        random_state=42
                    ).generate_from_frequencies(keywords_dict)
            except Exception as e:
                print(f"오류발생 {str(e)}")
            
            return wc
        
        except Exception as e:
            st.error(f"오류발생 {str(e)}")
            return None

# 뉴스 분석 함수
def analyze_news_content(news_df):
    if news_df.empty:
        return "데이터가 없습니다"
    
    results = {}
    #카테고리별
    if 'source' in news_df.columns:
            results['source_counts'] = news_df['source'].value_counts().to_dict()
    #카테고리별
    if 'date' in news_df.columns:
            results['date_counts'] = news_df['date'].value_counts().to_dict()

    #키워드분석
    all_text = " ".join(news_df['title'].fillna('') + " " + news_df['content'].fillna(''))

    if len(all_text.strip()) > 0:
        results['top_keywords_for_wordcloud']= extract_keywords_for_wordcloud(all_text, top_n=50)
        results['top_keywords'] = analyze_keywords(all_text)
    else:
        results['top_keywords_for_wordcloud']={}
        results['top_keywords'] = []
    return results

# OpenAI API를 이용한 새 기사 생성
def generate_article(original_content, prompt_text):
    try:
        response = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "당신은 전문적인 뉴스 기자입니다. 주어진 내용을 바탕으로 새로운 기사를 작성해주세요."},
                {"role": "user", "content": f"다음 내용을 바탕으로 {prompt_text}\n\n{original_content[:1000]}"}
            ],
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"기사 생성 오류: {str(e)}"

# OpenAI API를 이용한 이미지 생성
def generate_image(prompt):
    try:
        response = openai.images.generate(
            model="gpt-image-1",
            prompt=prompt
        )
        image_base64=response.data[0].b64_json
        return f"data:image/png;base64,{image_base64}"
    except Exception as e:
        return f"이미지 생성 오류: {str(e)}"

# 스케줄러 관련 함수들
def get_next_run_time(hour, minute):
    now = datetime.now()
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if next_run <= now:
        next_run += timedelta(days=1)
    return next_run

def run_scheduled_task():
    try:
        while global_scheduler_state.is_running:
            schedule.run_pending()
            time.sleep(1)
    except Exception as e:
        print(f"스케줄러 에러 발생: {e}")
        traceback.print_exc()

def perform_news_task(task_type, keyword, num_articles, file_prefix):
    try:
        articles = crawl_naver_news(keyword, num_articles)
        
        # 기사 내용 가져오기
        for article in articles:
            article['content'] = get_article_content(article['link'])
            time.sleep(0.5)  # 서버 부하 방지
        
        # 결과 저장
        os.makedirs('scheduled_news', exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"scheduled_news/{file_prefix}_{task_type}_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        global_scheduler_state.last_run = datetime.now()
        print(f"{datetime.now()} - {task_type} 뉴스 기사 수집 완료: {keyword}")
        
        # 전역 상태에 수집 결과를 저장 (UI 업데이트용)
        result_item = {
            'task_type': task_type,
            'keyword': keyword,
            'timestamp': timestamp,
            'num_articles': len(articles),
            'filename': filename
        }
        global_scheduler_state.scheduled_results.append(result_item)
        
    except Exception as e:
        print(f"작업 실행 중 오류 발생: {e}")
        traceback.print_exc()

def start_scheduler(daily_tasks, interval_tasks):
    if not global_scheduler_state.is_running:
        schedule.clear()
        global_scheduler_state.scheduled_jobs = []
        
        # 일별 태스크 등록
        for task in daily_tasks:
            hour = task['hour']
            minute = task['minute']
            keyword = task['keyword']
            num_articles = task['num_articles']
            
            job_id = f"daily_{keyword}_{hour}_{minute}"
            schedule.every().day.at(f"{hour:02d}:{minute:02d}").do(
                perform_news_task, "daily", keyword, num_articles, job_id
            ).tag(job_id)
            
            global_scheduler_state.scheduled_jobs.append({
                'id': job_id,
                'type': 'daily',
                'time': f"{hour:02d}:{minute:02d}",
                'keyword': keyword,
                'num_articles': num_articles
            })
        
        # 시간 간격 태스크 등록
        for task in interval_tasks:
            interval_minutes = task['interval_minutes']
            keyword = task['keyword']
            num_articles = task['num_articles']
            run_immediately = task['run_immediately']
            
            job_id = f"interval_{keyword}_{interval_minutes}"
            
            if run_immediately:
                # 즉시 실행
                perform_news_task("interval", keyword, num_articles, job_id)
            
            # 분 간격으로 예약
            schedule.every(interval_minutes).minutes.do(
                perform_news_task, "interval", keyword, num_articles, job_id
            ).tag(job_id)
            
            global_scheduler_state.scheduled_jobs.append({
                'id': job_id,
                'type': 'interval',
                'interval': f"{interval_minutes}분마다",
                'keyword': keyword,
                'num_articles': num_articles,
                'run_immediately': run_immediately
            })
        
        # 다음 실행 시간 계산
        next_run = schedule.next_run()
        if next_run:
            global_scheduler_state.next_run = next_run
        
        # 스케줄러 쓰레드 시작
        global_scheduler_state.is_running = True
        global_scheduler_state.thread = threading.Thread(
            target=run_scheduled_task, daemon=True
        )
        global_scheduler_state.thread.start()
        
        # 상태를 세션 상태로도 복사 (UI 표시용)
        if 'scheduler_status' not in st.session_state:
            st.session_state.scheduler_status = {}
        
        st.session_state.scheduler_status = {
            'is_running': global_scheduler_state.is_running,
            'last_run': global_scheduler_state.last_run,
            'next_run': global_scheduler_state.next_run,
            'jobs_count': len(global_scheduler_state.scheduled_jobs)
        }

def stop_scheduler():
    if global_scheduler_state.is_running:
        global_scheduler_state.is_running = False
        schedule.clear()
        if global_scheduler_state.thread:
            global_scheduler_state.thread.join(timeout=1)
        global_scheduler_state.next_run = None
        global_scheduler_state.scheduled_jobs = []
        
        # UI 상태 업데이트
        if 'scheduler_status' in st.session_state:
            st.session_state.scheduler_status['is_running'] = False

# 메뉴에 따른 화면 표시
if menu == "뉴스 기사 크롤링":
    st.header("뉴스 기사 크롤링")
    
    keyword = st.text_input("검색어 입력", "인공지능")
    num_articles = st.slider("가져올 기사 수", min_value=1, max_value=20, value=5)
    
    if st.button("기사 가져오기"):
        with st.spinner("기사를 수집 중입니다..."):
            articles = crawl_naver_news(keyword, num_articles)
            
            # 기사 내용 가져오기
            for i, article in enumerate(articles):
                st.progress((i + 1) / len(articles))
                article['content'] = get_article_content(article['link'])
                time.sleep(0.5)  # 서버 부하 방지
            
            # 결과 저장 및 표시
            save_articles(articles)
            st.success(f"{len(articles)}개의 기사를 수집했습니다!")
            
            # 수집한 기사 표시
            for article in articles:
                with st.expander(f"{article['title']} - {article['source']}"):
                    st.write(f"**출처:** {article['source']}")
                    st.write(f"**날짜:** {article['date']}")
                    st.write(f"**요약:** {article['description']}")
                    st.write(f"**링크:** {article['link']}")
                    st.write("**본문 미리보기:**")
                    st.write(article['content'][:300] + "...")

elif menu == "기사 분석하기":
    st.header("기사 분석하기")
    
    articles = load_saved_articles()
    if not articles:
        st.warning("저장된 기사가 없습니다. 먼저 '뉴스 기사 크롤링' 메뉴에서 기사를 수집해주세요.")
    else:
        # 기사 선택
        titles = [article['title'] for article in articles]
        selected_title = st.selectbox("분석할 기사 선택", titles)
        
        selected_article = next((a for a in articles if a['title'] == selected_title), None)
        
        if selected_article:
            st.write(f"**제목:** {selected_article['title']}")
            st.write(f"**출처:** {selected_article['source']}")
            
            # 본문 표시
            with st.expander("기사 본문 보기"):
                st.write(selected_article['content'])
            
            # 분석 방법 선택
            analysis_type = st.radio(
                "분석 방법",
                ["키워드 분석", "감정 분석", "텍스트 통계"]
            )
            
            if analysis_type == "키워드 분석":
                if st.button("키워드 분석하기"):
                    with st.spinner("키워드를 분석 중입니다..."):
                        keyword_tab1, keyword_tab2 = st.tabs(["키워드 빈도", "워드클라우드"])

                        with keyword_tab1:

                            keywords = analyze_keywords(selected_article['content'])
                            
                            # 시각화
                            df = pd.DataFrame(keywords, columns=['단어', '빈도수'])
                            st.bar_chart(df.set_index('단어'))
                            
                            st.write("**주요 키워드:**")
                            for word, count in keywords:
                                st.write(f"- {word}: {count}회")
                        with keyword_tab2:
                            keyword_dict = extract_keywords_for_wordcloud(selected_article['content'])
                            wc = generate_wordcloud(keyword_dict)
                            
                            if wc:
                                fig, ax = plt.subplots(figsize=(10, 5))
                                ax.imshow(wc, interpolation='bilinear')
                                ax.axis('off')
                                st.pyplot(fig)
                                
                                # 키워드 상위 20개 표시
                                st.write("**상위 20개 키워드:**")
                                top_keywords = sorted(keyword_dict.items(), key=lambda x: x[1], reverse=True)[:20]
                                keyword_df = pd.DataFrame(top_keywords, columns=['키워드', '빈도'])
                                st.dataframe(keyword_df)
                            else:
                                st.error("워드클라우드를 생성할 수 없습니다.")

            elif analysis_type == "텍스트 통계":
                if st.button("텍스트 통계 분석"):
                    content = selected_article['content']
                    
                    # 텍스트 통계 계산
                    word_count = len(re.findall(r'\b\w+\b', content))
                    char_count = len(content)
                    sentence_count = len(re.split(r'[.!?]+', content))
                    avg_word_length = sum(len(word) for word in re.findall(r'\b\w+\b', content)) / word_count if word_count > 0 else 0
                    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
                    
                    # 통계 표시
                    st.subheader("텍스트 통계")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("단어 수", f"{word_count:,}")
                    with col2:
                        st.metric("문자 수", f"{char_count:,}")
                    with col3:
                        st.metric("문장 수", f"{sentence_count:,}")
                        
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("평균 단어 길이", f"{avg_word_length:.1f}자")
                    with col2:
                        st.metric("평균 문장 길이", f"{avg_sentence_length:.1f}단어")
                    
                    # 텍스트 복잡성 점수 (간단한 예시)
                    complexity_score = min(10, (avg_sentence_length / 10) * 5 + (avg_word_length / 5) * 5)
                    st.progress(complexity_score / 10)
                    st.write(f"텍스트 복잡성 점수: {complexity_score:.1f}/10")
                        
                    # 출현 빈도 막대 그래프
                    st.subheader("품사별 분포 (한국어/영어 지원)")
                    try:
                        # KoNLPy 설치 확인
                        try:
                            from konlpy.tag import Okt
                            konlpy_installed = True
                        except ImportError:
                            konlpy_installed = False
                            st.warning("한국어 형태소 분석을 위해 KoNLPy를 설치해주세요: pip install konlpy")
                        
                        # 영어 POS tagger 준비
                        from nltk import pos_tag
                        try:
                            nltk.data.find('taggers/averaged_perceptron_tagger')
                        except LookupError:
                            nltk.download('averaged_perceptron_tagger')
                        
                        # Try using the correct resource name as shown in the error message
                        try:
                            nltk.data.find('averaged_perceptron_tagger_eng')
                        except LookupError:
                            nltk.download('averaged_perceptron_tagger_eng')
                        
                        # 언어 감지 (간단한 방식)
                        is_korean = bool(re.search(r'[가-힣]', content))
                        
                        if is_korean and konlpy_installed:
                            # 한국어 형태소 분석
                            okt = Okt()
                            tagged = okt.pos(content)
                            
                            # 한국어 품사 매핑
                            pos_dict = {
                                'Noun': '명사', 'NNG': '명사', 'NNP': '고유명사', 
                                'Verb': '동사', 'VV': '동사', 'VA': '형용사',
                                'Adjective': '형용사', 
                                'Adverb': '부사',
                                'Josa': '조사', 'Punctuation': '구두점',
                                'Determiner': '관형사', 'Exclamation': '감탄사'
                            }
                            
                            pos_counts = {'명사': 0, '동사': 0, '형용사': 0, '부사': 0, '조사': 0, '구두점': 0, '관형사': 0, '감탄사': 0, '기타': 0}
                            
                            for _, pos in tagged:
                                if pos in pos_dict:
                                    pos_counts[pos_dict[pos]] += 1
                                elif pos.startswith('N'):  # 기타 명사류
                                    pos_counts['명사'] += 1
                                elif pos.startswith('V'):  # 기타 동사류
                                    pos_counts['동사'] += 1
                                else:
                                    pos_counts['기타'] += 1
                                    
                        else:
                            # 영어 POS 태깅
                            tokens = word_tokenize(content.lower())
                            tagged = pos_tag(tokens)
                            
                            # 영어 품사 매핑
                            pos_dict = {
                                'NN': '명사', 'NNS': '명사', 'NNP': '고유명사', 'NNPS': '고유명사', 
                                'VB': '동사', 'VBD': '동사', 'VBG': '동사', 'VBN': '동사', 'VBP': '동사', 'VBZ': '동사',
                                'JJ': '형용사', 'JJR': '형용사', 'JJS': '형용사',
                                'RB': '부사', 'RBR': '부사', 'RBS': '부사'
                            }
                            
                            pos_counts = {'명사': 0, '동사': 0, '형용사': 0, '부사': 0, '기타': 0}
                            
                            for _, pos in tagged:
                                if pos in pos_dict:
                                    pos_counts[pos_dict[pos]] += 1
                                else:
                                    pos_counts['기타'] += 1
                        
                        # 결과 시각화
                        pos_df = pd.DataFrame({
                            '품사': list(pos_counts.keys()),
                            '빈도': list(pos_counts.values())
                        })
                        
                        st.bar_chart(pos_df.set_index('품사'))
                        
                        if is_korean:
                            st.info("한국어 텍스트가 감지되었습니다.")
                        else:
                            st.info("영어 텍스트가 감지되었습니다.")
                    except Exception as e:
                        st.error(f"품사 분석 중 오류 발생: {str(e)}")
                        st.error(traceback.format_exc())

            elif analysis_type == "감정 분석":
                if st.button("감정 분석하기"):
                    if st.session_state.openai_api_key:
                        with st.spinner("기사의 감정을 분석 중입니다..."):
                            try:
                                openai.api_key = st.session_state.openai_api_key
                                
                                # 감정 분석 프롬프트 설정
                                response = openai.chat.completions.create(
                                    model="gpt-4.1-mini",
                                    messages=[
                                        {"role": "system", "content": "당신은 텍스트의 감정과 논조를 분석하는 전문가입니다. 다음 뉴스 기사의 감정과 논조를 분석하고, '긍정적', '부정적', '중립적' 중 하나로 분류해 주세요. 또한 기사에서 드러나는 핵심 감정 키워드를 5개 추출하고, 각 키워드별로 1-10 사이의 강도 점수를 매겨주세요. JSON 형식으로 다음과 같이 응답해주세요: {'sentiment': '긍정적/부정적/중립적', 'reason': '이유 설명...', 'keywords': [{'word': '키워드1', 'score': 8}, {'word': '키워드2', 'score': 7}, ...]}"},
                                        {"role": "user", "content": f"다음 뉴스 기사를 분석해 주세요:\n\n제목: {selected_article['title']}\n\n내용: {selected_article['content'][:1500]}"}
                                    ],
                                    max_tokens=800,
                                    response_format={"type": "json_object"}
                                )
                                
                                # JSON 파싱
                                analysis_result = json.loads(response.choices[0].message.content)
                                
                                # 결과 시각화
                                st.subheader("감정 분석 결과")
                                
                                # 1. 감정 타입에 따른 시각적 표현
                                sentiment_type = analysis_result.get('sentiment', '중립적')
                                col1, col2, col3 = st.columns([1, 3, 1])
                                
                                with col2:
                                    if sentiment_type == "긍정적":
                                        st.markdown(f"""
                                        <div style="background-color:#DCEDC8; padding:20px; border-radius:10px; text-align:center;">
                                            <h1 style="color:#388E3C; font-size:28px;">😀 긍정적 논조 😀</h1>
                                            <p style="font-size:16px;">감정 강도: 높음</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    elif sentiment_type == "부정적":
                                        st.markdown(f"""
                                        <div style="background-color:#FFCDD2; padding:20px; border-radius:10px; text-align:center;">
                                            <h1 style="color:#D32F2F; font-size:28px;">😞 부정적 논조 😞</h1>
                                            <p style="font-size:16px;">감정 강도: 높음</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    else:
                                        st.markdown(f"""
                                        <div style="background-color:#E0E0E0; padding:20px; border-radius:10px; text-align:center;">
                                            <h1 style="color:#616161; font-size:28px;">😐 중립적 논조 😐</h1>
                                            <p style="font-size:16px;">감정 강도: 중간</p>
                                        </div>
                                        """, unsafe_allow_html=True)
                                
                                # 2. 이유 설명
                                st.markdown("### 분석 근거")
                                st.markdown(f"<div style='background-color:#F5F5F5; padding:15px; border-radius:5px;'>{analysis_result.get('reason', '')}</div>", unsafe_allow_html=True)
                                
                                # 3. 감정 키워드 시각화
                                st.markdown("### 핵심 감정 키워드")
                                
                                # 키워드 데이터 준비
                                keywords = analysis_result.get('keywords', [])
                                if keywords:
                                    # 막대 차트용 데이터
                                    keyword_names = [item.get('word', '') for item in keywords]
                                    keyword_scores = [item.get('score', 0) for item in keywords]
                                    
                                    # 레이더 차트 생성
                                    fig = go.Figure()
                                    
                                    # 색상 설정
                                    if sentiment_type == "긍정적":
                                        fill_color = 'rgba(76, 175, 80, 0.3)'  # 연한 초록색
                                        line_color = 'rgba(76, 175, 80, 1)'     # 진한 초록색
                                    elif sentiment_type == "부정적":
                                        fill_color = 'rgba(244, 67, 54, 0.3)'   # 연한 빨간색
                                        line_color = 'rgba(244, 67, 54, 1)'     # 진한 빨간색
                                    else:
                                        fill_color = 'rgba(158, 158, 158, 0.3)' # 연한 회색
                                        line_color = 'rgba(158, 158, 158, 1)'   # 진한 회색
                                    
                                    # 레이더 차트 데이터 준비 - 마지막 점이 첫 점과 연결되도록 데이터 추가
                                    radar_keywords = keyword_names.copy()
                                    radar_scores = keyword_scores.copy()
                                    
                                    # 레이더 차트 생성
                                    fig.add_trace(go.Scatterpolar(
                                        r=radar_scores,
                                        theta=radar_keywords,
                                        fill='toself',
                                        fillcolor=fill_color,
                                        line=dict(color=line_color, width=2),
                                        name='감정 키워드'
                                    ))
                                    
                                    # 레이더 차트 레이아웃 설정
                                    fig.update_layout(
                                        polar=dict(
                                            radialaxis=dict(
                                                visible=True,
                                                range=[0, 10],
                                                tickmode='linear',
                                                tick0=0,
                                                dtick=2
                                            )
                                        ),
                                        showlegend=False,
                                        title={
                                            'text': '감정 키워드 레이더 분석',
                                            'y':0.95,
                                            'x':0.5,
                                            'xanchor': 'center',
                                            'yanchor': 'top'
                                        },
                                        height=500,
                                        width=500,
                                        margin=dict(l=80, r=80, t=80, b=80)
                                    )
                                    
                                    # 차트 중앙에 표시
                                    col1, col2, col3 = st.columns([1, 2, 1])
                                    with col2:
                                        st.plotly_chart(fig)
                                    
                                    # 키워드 카드로 표시
                                    st.markdown("#### 키워드 세부 설명")
                                    cols = st.columns(min(len(keywords), 5))
                                    for i, keyword in enumerate(keywords):
                                        with cols[i % len(cols)]:
                                            word = keyword.get('word', '')
                                            score = keyword.get('score', 0)
                                            
                                            # 점수에 따른 색상 계산
                                            r, g, b = 0, 0, 0
                                            if sentiment_type == "긍정적":
                                                g = min(200 + score * 5, 255)
                                                r = max(255 - score * 20, 100)
                                            elif sentiment_type == "부정적":
                                                r = min(200 + score * 5, 255)
                                                g = max(255 - score * 20, 100)
                                            else:
                                                r = g = b = 128
                                            
                                            # 카드 생성
                                            st.markdown(f"""
                                            <div style="background-color:rgba({r},{g},{b},0.2); padding:10px; border-radius:5px; text-align:center; margin:5px;">
                                                <h3 style="margin:0;">{word}</h3>
                                                <div style="background-color:#E0E0E0; border-radius:3px; margin-top:5px;">
                                                    <div style="width:{score*10}%; background-color:rgba({r},{g},{b},0.8); height:10px; border-radius:3px;"></div>
                                                </div>
                                                <p style="margin:2px; font-size:12px;">강도: {score}/10</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    
                                else:
                                    st.info("키워드를 추출하지 못했습니다.")
                                    
                                # 4. 요약 통계
                                st.markdown("### 주요 통계")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(label="긍정/부정 점수", value=f"{7 if sentiment_type == '긍정적' else 3 if sentiment_type == '부정적' else 5}/10")
                                with col2:
                                    st.metric(label="키워드 수", value=len(keywords))
                                with col3:
                                    avg_score = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0
                                    st.metric(label="평균 강도", value=f"{avg_score:.1f}/10")
                                
                            except Exception as e:
                                st.error(f"감정 분석 오류: {str(e)}")
                                st.code(traceback.format_exc())
                    else:
                        st.warning("OpenAI API 키가 설정되어 있지 않습니다. 사이드바에서 API 키를 설정해주세요.")

elif menu == "새 기사 생성하기":
    st.header("새 기사 생성하기")
    
    articles = load_saved_articles()
    if not articles:
        st.warning("저장된 기사가 없습니다. 먼저 '뉴스 기사 크롤링' 메뉴에서 기사를 수집해주세요.")
    else:
        # 기사 선택
        titles = [article['title'] for article in articles]
        selected_title = st.selectbox("원본 기사 선택", titles)
        
        selected_article = next((a for a in articles if a['title'] == selected_title), None)
        
        if selected_article:
            st.write(f"**원본 제목:** {selected_article['title']}")
            
            with st.expander("원본 기사 내용"):
                st.write(selected_article['content'])
            
            prompt_text ="""다음 기사 양식을 따라서 다시 작성해줘. 
역할: 당신은 신문사의 기자입니다.
작업: 최근 일어난 사건에 대한 보도자료를 작성해야 합니다. 자료는 사실을 기반으로 하며, 객관적이고 정확해야 합니다.
지침:
제공된 정보를 바탕으로 신문 보도자료 형식에 맞춰 기사를 작성하세요.
기사 제목은 주제를 명확히 반영하고 독자의 관심을 끌 수 있도록 작성합니다.
기사 내용은 정확하고 간결하며 설득력 있는 문장으로 구성합니다.
관련자의 인터뷰를 인용 형태로 넣어주세요.
위의 정보와 지침을 참고하여 신문 보도자료 형식의 기사를 작성해 주세요"""
            
            # 이미지 생성 여부 선택 옵션 추가
            generate_image_too = st.checkbox("기사 생성 후 이미지도 함께 생성하기", value=True)
            
            if st.button("새 기사 생성하기"):
                if st.session_state.openai_api_key:
                    openai.api_key = st.session_state.openai_api_key
                    with st.spinner("기사를 생성 중입니다..."):
                        new_article = generate_article(selected_article['content'], prompt_text)
                        
                        st.write("**생성된 기사:**")
                        st.write(new_article)
                        
                        # 이미지 생성하기 (옵션이 선택된 경우)
                        if generate_image_too:
                            with st.spinner("기사 관련 이미지를 생성 중입니다..."):
                                # 이미지 생성 프롬프트 준비
                                image_prompt = f"""신문기사 제목 "{selected_article['title']}" 을 보고 이미지를 만들어줘 
                                이미지에는 다음 요소가 포함되어야 합니다:
                                - 기사를 이해할 수 있는 도식
                                - 기사 내용과 관련된 텍스트 
                                - 심플하게 처리
                                """
                                
                                # 이미지 생성
                                image_url = generate_image(image_prompt)
                                
                                if image_url and not image_url.startswith("이미지 생성 오류"):
                                    st.subheader("생성된 이미지:")
                                    st.image(image_url)
                                else:
                                    st.error(image_url)
                        
                        # 생성된 기사 저장 옵션
                        if st.button("생성된 기사 저장"):
                            new_article_data = {
                                'title': f"[생성됨] {selected_article['title']}",
                                'source': f"AI 생성 (원본: {selected_article['source']})",
                                'date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                                'description': new_article[:100] + "...",
                                'link': "",
                                'content': new_article
                            }
                            articles.append(new_article_data)
                            save_articles(articles)
                            st.success("생성된 기사가 저장되었습니다!")
                else:
                    st.warning("OpenAI API 키를 사이드바에서 설정해주세요.")



elif menu == "뉴스 기사 예약하기":
    st.header("뉴스 기사 예약하기")
    
    # 탭 생성
    tab1, tab2, tab3 = st.tabs(["일별 예약", "시간 간격 예약", "스케줄러 상태"])
    
    # 일별 예약 탭
    with tab1:
        st.subheader("매일 정해진 시간에 기사 수집하기")
        
        # 키워드 입력
        daily_keyword = st.text_input("검색 키워드", value="인공지능", key="daily_keyword")
        daily_num_articles = st.slider("수집할 기사 수", min_value=1, max_value=20, value=5, key="daily_num_articles")
        
        # 시간 설정
        daily_col1, daily_col2 = st.columns(2)
        with daily_col1:
            daily_hour = st.selectbox("시", range(24), format_func=lambda x: f"{x:02d}시", key="daily_hour")
        with daily_col2:
            daily_minute = st.selectbox("분", range(0, 60, 5), format_func=lambda x: f"{x:02d}분", key="daily_minute")
        
        # 일별 예약 리스트
        if 'daily_tasks' not in st.session_state:
            st.session_state.daily_tasks = []
        
        if st.button("일별 예약 추가"):
            st.session_state.daily_tasks.append({
                'hour': daily_hour,
                'minute': daily_minute,
                'keyword': daily_keyword,
                'num_articles': daily_num_articles
            })
            st.success(f"일별 예약이 추가되었습니다: 매일 {daily_hour:02d}:{daily_minute:02d} - '{daily_keyword}'")
        
        # 예약 목록 표시
        if st.session_state.daily_tasks:
            st.subheader("일별 예약 목록")
            for i, task in enumerate(st.session_state.daily_tasks):
                st.write(f"{i+1}. 매일 {task['hour']:02d}:{task['minute']:02d} - '{task['keyword']}' ({task['num_articles']}개)")
            
            if st.button("일별 예약 초기화"):
                st.session_state.daily_tasks = []
                st.warning("일별 예약이 모두 초기화되었습니다.")
    
    # 시간 간격 예약 탭
    with tab2:
        st.subheader("시간 간격으로 기사 수집하기")
        
        # 키워드 입력
        interval_keyword = st.text_input("검색 키워드", value="빅데이터", key="interval_keyword")
        interval_num_articles = st.slider("수집할 기사 수", min_value=1, max_value=20, value=5, key="interval_num_articles")
        
        # 시간 간격 설정
        interval_minutes = st.number_input("실행 간격(분)", min_value=1, max_value=60*24, value=30, key="interval_minutes")
        
        # 즉시 실행 여부
        run_immediately = st.checkbox("즉시 실행", value=True, help="체크하면 스케줄러 시작 시 즉시 실행합니다.")
        
        # 시간 간격 예약 리스트
        if 'interval_tasks' not in st.session_state:
            st.session_state.interval_tasks = []
        
        if st.button("시간 간격 예약 추가"):
            st.session_state.interval_tasks.append({
                'interval_minutes': interval_minutes,
                'keyword': interval_keyword,
                'num_articles': interval_num_articles,
                'run_immediately': run_immediately
            })
            st.success(f"시간 간격 예약이 추가되었습니다: {interval_minutes}분마다 - '{interval_keyword}'")
        
        # 예약 목록 표시
        if st.session_state.interval_tasks:
            st.subheader("시간 간격 예약 목록")
            for i, task in enumerate(st.session_state.interval_tasks):
                immediate_text = "즉시 실행 후 " if task['run_immediately'] else ""
                st.write(f"{i+1}. {immediate_text}{task['interval_minutes']}분마다 - '{task['keyword']}' ({task['num_articles']}개)")
            
            if st.button("시간 간격 예약 초기화"):
                st.session_state.interval_tasks = []
                st.warning("시간 간격 예약이 모두 초기화되었습니다.")
    
    # 스케줄러 상태 탭
    with tab3:
        st.subheader("스케줄러 제어 및 상태")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 스케줄러 시작/중지 버튼
            if not global_scheduler_state.is_running:
                if st.button("스케줄러 시작"):
                    if not st.session_state.daily_tasks and not st.session_state.interval_tasks:
                        st.error("예약된 작업이 없습니다. 먼저 일별 예약 또는 시간 간격 예약을 추가해주세요.")
                    else:
                        start_scheduler(st.session_state.daily_tasks, st.session_state.interval_tasks)
                        st.success("스케줄러가 시작되었습니다.")
            else:
                if st.button("스케줄러 중지"):
                    stop_scheduler()
                    st.warning("스케줄러가 중지되었습니다.")
        
        with col2:
            # 스케줄러 상태 표시
            if 'scheduler_status' in st.session_state:
                st.write(f"상태: {'실행중' if global_scheduler_state.is_running else '중지'}")
                if global_scheduler_state.last_run:
                    st.write(f"마지막 실행: {global_scheduler_state.last_run.strftime('%Y-%m-%d %H:%M:%S')}")
                if global_scheduler_state.next_run and global_scheduler_state.is_running:
                    st.write(f"다음 실행: {global_scheduler_state.next_run.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                st.write("상태: 중지")
        
        # 예약된 작업 목록
        if global_scheduler_state.scheduled_jobs:
            st.subheader("현재 실행 중인 예약 작업")
            for i, job in enumerate(global_scheduler_state.scheduled_jobs):
                if job['type'] == 'daily':
                    st.write(f"{i+1}. [일별] 매일 {job['time']} - '{job['keyword']}' ({job['num_articles']}개)")
                else:
                    immediate_text = "[즉시 실행 후] " if job.get('run_immediately', False) else ""
                    st.write(f"{i+1}. [간격] {immediate_text}{job['interval']} - '{job['keyword']}' ({job['num_articles']}개)")
        
        # 스케줄러 실행 결과
        if global_scheduler_state.scheduled_results:
            st.subheader("스케줄러 실행 결과")
            
            # 결과를 UI에 표시하기 전에 복사
            results_for_display = global_scheduler_state.scheduled_results.copy()
            
            if results_for_display:
                result_df = pd.DataFrame(results_for_display)
                result_df['실행시간'] = result_df['timestamp'].apply(lambda x: datetime.strptime(x, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S"))
                result_df = result_df.rename(columns={
                    'task_type': '작업유형', 
                    'keyword': '키워드', 
                    'num_articles': '기사수', 
                    'filename': '파일명'
                })
                result_df['작업유형'] = result_df['작업유형'].apply(lambda x: '일별' if x == 'daily' else '시간간격')
                
                st.dataframe(
                    result_df[['작업유형', '키워드', '기사수', '실행시간', '파일명']],
                    hide_index=True
                )
        
        # 수집된 파일 보기
        if os.path.exists('scheduled_news'):
            files = [f for f in os.listdir('scheduled_news') if f.endswith('.json')]
            if files:
                st.subheader("수집된 파일 열기")
                selected_file = st.selectbox("파일 선택", files, index=len(files)-1)
                if selected_file and st.button("파일 내용 보기"):
                    with open(os.path.join('scheduled_news', selected_file), 'r', encoding='utf-8') as f:
                        articles = json.load(f)
                    
                    st.write(f"**파일명:** {selected_file}")
                    st.write(f"**수집 기사 수:** {len(articles)}개")
                    
                    for article in articles:
                        with st.expander(f"{article['title']} - {article['source']}"):
                            st.write(f"**출처:** {article['source']}")
                            st.write(f"**날짜:** {article['date']}")
                            st.write(f"**링크:** {article['link']}")
                            st.write("**본문:**")
                            st.write(article['content'][:500] + "..." if len(article['content']) > 500 else article['content'])

# 푸터
st.markdown("---")
st.markdown("© 뉴스 기사 도구 @conanssam")
