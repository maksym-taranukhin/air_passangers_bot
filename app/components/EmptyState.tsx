import { Heading } from "@chakra-ui/react";

export function EmptyState(props: { onChoice: (question: string) => any }) {
  return (
    <div className="rounded flex flex-col items-center max-w-full md:p-8">
      <Heading fontSize="3xl" fontWeight={"medium"} mb={1} color={"black"}>Air Passenger Rights Chatbot ✈️ (DEMO)</Heading>
      <Heading fontSize="xl" fontWeight={"normal"} mb={1} color={"black"} marginTop={"10px"} textAlign={"center"}>
        Your Ticket to Clear Answers - Just Ask!{" "}
      </Heading>
    </div>
  );
}
