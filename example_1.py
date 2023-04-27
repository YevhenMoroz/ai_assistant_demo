from chronological import read_prompt, main, cleaned_completion


async def generate_dma_order_code(params):
    prompt = read_prompt('order_creation_prompt')
    return await cleaned_completion(
        prompt.format(params),
        engine="davinci",
        temperature=0.7,
        max_tokens=300,
        stop=["\n\n"],
    )


async def logic():
    result = await generate_dma_order_code({"Price": "10", "Qty": "500", "Account": "testAcc"})
    # result = await generate_dma_order_code("without change_parameter")
    print(result)


main(logic)
