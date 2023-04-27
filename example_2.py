from chronological import read_prompt, main, cleaned_completion


async def generate_code_description(params):
    prompt = read_prompt('code_description')
    return await cleaned_completion(
        prompt.format(params),
        engine="davinci",
        temperature=0.2,
        max_tokens=300,
        stop=["\n\n"],
    )


async def logic():
    favorite_foods = await generate_code_description("self.fix_manager = FixManager(self.fix_env.sell_side, self.test_id)\n        self.fix_message = FixMessageNewOrderSingleOMS(self.data_set)\n        self.fix_message.set_default_dma_limit()\n        self.fix_message.change_parameter(\"Price\",\"10\")\n        self.fix_message.change_parameter(\"Qty\",\"500\")\n        self.fix_message.change_parameter(\"Account\",\"testAcc\")\n        self.fix_manager.send_message_fix_standard(self.fix_message)")
    print(favorite_foods)


main(logic)
