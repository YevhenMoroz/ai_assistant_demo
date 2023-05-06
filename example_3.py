from chronological import read_prompt, main, cleaned_chat_completion


async def generate_dma_order_code(params):
    prompt = read_prompt('order_creation_prompt')
    return await cleaned_chat_completion(
        prompt.format(params),
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=300,
        stop=["\n\n"],
    )


async def generate_verifier(params):
    prompt = read_prompt('verifier_prompt')
    return await cleaned_chat_completion(
        prompt.format(params),
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=300,
        stop=["\n\n"],
    )


async def generate_test(test_number, params):
    prompt = read_prompt('test_generation_prompt')
    return await cleaned_chat_completion(
        prompt.format(test_number, params),
        model="gpt-3.5-turbo",
        temperature=0.2,
        max_tokens=3400,
        stop=["\n\n"],
    )


async def logic():
    order_block = await generate_dma_order_code("without changing parameters")
    verifier_block = await generate_verifier("for new order on buy side")
    test = await generate_test("7318", order_block + verifier_block)
    print(test)


main(logic)
