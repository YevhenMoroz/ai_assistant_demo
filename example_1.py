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


async def logic():
    # result = await generate_dma_order_code({"Price": "10", "Qty": "500", "Account": "testAcc"})
    result = await generate_dma_order_code("without changing parameters")
    print(result)


main(logic)
