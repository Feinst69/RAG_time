import dspy
from rag_time.config.settings import settings
from rag_time.data_loader import RetrievedTicketsDict


def init_lm() -> dspy.LM:
    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required to translate tickets")

    model = settings.openrouter_model.strip()
    if not model:
        raise RuntimeError("OPENROUTER_MODEL is required to translate tickets")
    if not model.startswith("openrouter/"):
        model = f"openrouter/{model}"

    return dspy.LM(
        api_key=settings.openrouter_api_key,
        api_base=settings.openrouter_api_base,
        model=model,
        temperature=settings.openrouter_temperature,
        timeout=settings.openrouter_timeout
    )

class TicketTranslationSignature(dspy.Signature):
    """
    Translate only the subject, body and answer fields of each support ticket into the same language as the query.
    The other fields, including the language field, must be kept unchanged. 
    """
    query: str = dspy.InputField(desc="query in the target language")
    tickets_in: RetrievedTicketsDict = dspy.InputField(desc="original dictionary of support tickets")
    tickets_out: RetrievedTicketsDict = dspy.OutputField(desc="dictionary of translated support tickets")


def translate_tickets(query: str, tickets_in: RetrievedTicketsDict) -> RetrievedTicketsDict:
    lm = init_lm()
    with dspy.context(lm=lm):
        translator = dspy.Predict(signature=TicketTranslationSignature)
        tickets_out = translator(query=query, tickets_in=tickets_in)
        return tickets_out


if __name__ == "__main__":
    data = {
      "be7ad56a-7075-57e9-be14-820cb000ec68": {
        "score": 0.5,
        "subject": "Application Crash",
        "body": "Facing an application crash while using project management software on a 4K touchscreen monitor. The issue might be due to compatibility problems, graphics driver issues, or high resolution settings. Despite updating the drivers and adjusting the screen resolution, the problem still persists. Your assistance in resolving this matter would be greatly appreciated, as it is affecting the team's productivity. Please let me know if there's any additional information needed to investigate further.",
        "answer": "<name>, I'm sorry to hear about the application crash with the project management software on your 4K touchscreen monitor. Thank you for the troubleshooting steps you've already taken, such as updating the drivers and adjusting the screen resolution. We will investigate the issue further. Could you please provide the exact error message you receive during the application crashes and your computer's specifications? We need <acc_num> to look into this matter, so please contact <tel_num> to discuss the necessary details.",
        "type": "Problem",
        "queue": "Technical Support",
        "priority": "high",
        "language": "en"
      },
      "df7a8538-24af-5bbe-bf89-59be5518d242": {
        "score": 0.33333334,
        "subject": "Issue with Application Crash",
        "body": "Experienced an application crash while running the project management software on a 4K touchscreen monitor. The problem might be due to compatibility issues or graphics driver settings related to the high resolution. Despite attempting to update the drivers and adjust the screen resolution, the issue still persists. Your assistance in resolving this matter would be greatly appreciated, as it is affecting team productivity. Please inform me if there is any additional information needed to further investigate the issue.",
        "answer": "<name>, sorry to hear about the application crash with the project management software on a 4K touchscreen monitor. Thank you for the troubleshooting steps you've already taken, such as updating the drivers and adjusting the screen resolution. We will investigate the issue further. Please provide the exact error message you receive during the application crashes and share your computer's specifications. Please contact <tel_num> with account number <acc_num> to discuss any additional information needed.",
        "type": "Problem",
        "queue": "Technical Support",
        "priority": "high",
        "language": "en"
      }
    }

    data = RetrievedTicketsDict(root=data)
    query = "écran noir"
    tickets_traduits = translate_tickets(query, data)
    print(tickets_traduits.tickets_out.root)
