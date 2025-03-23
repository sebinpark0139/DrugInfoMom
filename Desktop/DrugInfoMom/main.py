import os
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated

# 상태 정의 (TypedDict 사용)
class GraphState(TypedDict):
    user_query: str
    csv_result: Annotated[str, "csv_result"]
    search_result: Annotated[str, "search_result"]
    final_response: str

# --- Node A: 사용자 입력 수집 노드 ---
def node_a_function(state):
    user_query = state["user_query"]
    return {"user_query": user_query}

# --- Node B: CSV 데이터 처리 ---
def node_b_function(state):
    query = state["user_query"]
    df = pd.read_csv('/Users/bagsebin/Desktop/DrugInfoMom/langgraph-example/임부금기 성분리스트.csv')
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="tool-calling",
        verbose=True,
        allow_dangerous_code=True
    )

    prompt = f"""
    CSV 데이터에서 '{query}'에 대한 정보를 정확히 찾고 답변하세요. CSV에 없으면 반드시 '해당 정보는 CSV에 없습니다.'라고 답변하세요.
    """

    result = agent_executor.invoke(prompt.strip())
    return {"csv_result": result}

# --- Node C: 웹 검색 노드 ---
def node_c_function(state):
    query = state["user_query"]
    tool = TavilySearchResults(max_results=2, tavily_api_key=os.getenv("TAVILY_API_KEY"))
    result = tool.invoke(query)
    return {"search_result": result}

# --- Node D: 결과 통합 및 최종 응답 생성 노드 ---
def node_d_function(state):
    user_query = state["user_query"]
    csv_result = state["csv_result"]
    search_result = state["search_result"]

    messages = [
        {"role": "system", "content": "CSV 데이터 결과와 웹 검색 결과를 참고하여 사용자의 질문에 정확히 답변하세요. CSV와 웹 검색 결과에서 서로 상충되거나 관련 없는 내용이 있으면 보다 신뢰할 수 있는 정보를 우선으로 답변을 작성하고, 자료 출처를 언급하지 마세요."},
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": f"CSV 데이터 결과:\n{csv_result}"},
        {"role": "assistant", "content": f"웹 검색 결과:\n{search_result}"},
        {"role": "user", "content": "위 내용을 정확히 종합하여 최종적으로 자연스러운 형태로 답변을 작성하세요."}
    ]

    llm = ChatOpenAI(model="gpt-4o-mini",temperature=0)
    final_response = llm.invoke(messages)
    return {"final_response": final_response.content}

# --- 상태 그래프 구성 ---
def build_graph():
    builder = StateGraph(GraphState)

    builder.add_node("a", node_a_function)
    builder.add_node("b", node_b_function)
    builder.add_node("c", node_c_function)
    builder.add_node("d", node_d_function)

    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("a", "c")
    builder.add_edge("b", "d")
    builder.add_edge("c", "d")
    builder.add_edge("d", END)

    return builder.compile()

if __name__ == "__main__":
    graph = build_graph()
    inputs = {"user_query": "나 34주야인데, 로라타딘 500mg짜리를 반 알 먹었어. 아기한테 문제가 없을까?"}
    final_state = graph.invoke(inputs)

    print("최종 응답:")
    print(final_state["final_response"])