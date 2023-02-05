from langchain.chains import LLMChain, ConversationChain
from langchain.prompts import PromptTemplate
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import SimpleSequentialChain, SequentialChain
from langchain.llms import OpenAI


def get_image_interpretation_chain(model='text-davinci-003', verbose=True):
    desc_llm = OpenAI(model_name=model, max_tokens=512)
    desc_prompt = PromptTemplate(
        input_variables=["image_descriptions", "ai_models"],
        template=(
            "I give you a list of descriptions of the same image by "
            "different AI models ({ai_models}), and you have to think of an "
            "association to that image. Here is the list of descriptions: "
            "\n"
            "{image_descriptions}"
            "\n"
            "Can you describe in detail how do you think this image looks "
            "like? Be specific. By the way, it might be not an image of "
            "the real world."))
    desc_chain = LLMChain(
        llm=desc_llm, prompt=desc_prompt,
        output_key="image_interpretation", verbose=verbose)
    return desc_chain


def get_generic_clue_chain(model='text-davinci-003', verbose=True):
    association_llm = OpenAI(model_name=model, max_tokens=512)
    association_prompt = PromptTemplate(
        input_variables=["image_interpretation", "qna_session"],
        template=(
            "For an image with a following detailed description:"
            "\n"
            "{image_interpretation}"
            "\n"
            "There was a following question-answering session:"
            "\n"
            "{qna_session}"
            "\n"
            "What does it associate with for you? Be abstract and creative! "
            "Any phylosophical thoughts? What does it remind you about?"))
    association_chain = LLMChain(
        llm=association_llm, prompt=association_prompt,
        output_key="association", verbose=verbose)

    clue_llm = OpenAI(model_name=model, max_tokens=512)
    clue_prompt = PromptTemplate(
        input_variables=["association"],
        template=(
            "Given the following association for an image"
            "\n"
            "{association}"
            "\n"
            "Summarize it in one short phrase, no more than 3 words:"
        ))
    clue_chain = LLMChain(
        llm=clue_llm, prompt=clue_prompt,
        output_key="clue", verbose=verbose)
    return SequentialChain(
        chains=[association_chain, clue_chain],
        input_variables=["image_interpretation", "qna_session"],
        output_variables=["clue", "association"],
        verbose=verbose)


def get_clue_chain(model='text-davinci-003',
                   personality='generic',
                   verbose=True):
    if personality == 'generic':
        return get_generic_clue_chain(model=model, verbose=verbose)


def talk_with_blip2(image_interpretation,
                    image,
                    ask_blip2_fn,
                    num_questions=2,
                    model='text-davinci-003',
                    verbose=True):
    image_interpretation = image_interpretation.strip()

    pre_llm = OpenAI(model_name=model, max_tokens=256)
    pre_prompt = PromptTemplate(
        input_variables=["image_interpretation"],
        template=(
            "You can't see this photo but you are given its short description:"
            "\n"
            "{image_interpretation}"
            "\n"
            "What additional information do you need to be able to tell "
            "a compelling story about what is happening in this photo?"))
    pre_chain = LLMChain(llm=pre_llm, prompt=pre_prompt, verbose=verbose)
    pre_results = pre_chain.predict(image_interpretation=image_interpretation)
    pre_results = pre_results.strip()

    llm = OpenAI(model_name=model, max_tokens=512)
    prompt = PromptTemplate(
        input_variables=["blip2_answer", "chat_history"],
        template=(
            "Your name is Bob, you're trying to talk with Alice to make sense "
            "of an photo that you don't see. Your task is to get more information "
            "about the photo from Alice by asking her short questions. "
            "Hint - a good question is about the actions happening in the "
            "photo and what a specific character is doing, or about other "
            "objects or living creatures that are present in the photo."
            "\n"
            "Alice: Here is the description of a photo that you don't see:"
            "\n"
            f"{image_interpretation.strip()}"
            "\n"
            f"Bob: {pre_results}"
            "\n"
            "{chat_history}"
            "\n"
            "Alice: {blip2_answer}"
            "\n"
            "Alice: You can ask me one more short question about the photo. "
            "\n"
            "Bob:"))
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        human_prefix="Alice",
        ai_prefix="Bob")
    chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=verbose)

    question_answering_log = []
    blip2_answer = "Only ask me questions that matters."
    for iter in range(num_questions):
        results = chain.predict(
            blip2_answer=blip2_answer.strip(),
        )
        if verbose:
            print('Question:', results)
        
        # Sometimes, the result can contain hallucinated answer so we should
        # filter it out:
        results = results[:results.find("Alice:")]
        question_answering_log.append("Question: " + results.strip())
        blip2_answer = ask_blip2_fn(image, results.strip())
        if verbose:
            print("Answer:", blip2_answer)
        question_answering_log.append("Answer: " + blip2_answer.strip())

    return '\n'.join(question_answering_log)


def make_sense_about_image(image,
                           generate_captions_fn,
                           ask_blip2_fn,
                           personality='generic',
                           openai_model='text-davinci-003',
                           num_blip2_questions=3,
                           verbose=True):
    # Generate deep captions
    captioning_results = generate_captions_fn(image)

    # Get first interpretation
    image_interp_chain = get_image_interpretation_chain(
        model=openai_model, verbose=verbose)
    image_interpretation = image_interp_chain.predict(
        image_descriptions=captioning_results["captions"],
        ai_models=", ".join(captioning_results["models"]))
    image_interpretation = image_interpretation.strip()

    # Talk with BLIP-v2 to get more information
    blip2_results = talk_with_blip2(
        image_interpretation=image_interpretation,
        image=image,
        num_questions=num_blip2_questions,
        model=openai_model,
        verbose=verbose)
    blip2_results = blip2_results.strip()

    # Generate clue
    clue_chain = get_clue_chain(
        model=openai_model,
        personality='generic',
        verbose=verbose)
    clue_results = clue_chain({
        'image_interpretation': image_interpretation,
        'qna_session': blip2_results,
    })
    return {
        'image_interpretation': image_interpretation,
        'qna_session': blip2_results,
        'association': clue_results['association'],
        'clue': clue_results['clue'],
    }
