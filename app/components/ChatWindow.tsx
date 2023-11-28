"use client";

// Importing necessary libraries and components
import React, { useRef, useState } from "react";
import { v4 as uuidv4 } from "uuid";
import { EmptyState } from "../components/EmptyState";
import { ChatMessageBubble, Message } from "../components/ChatMessageBubble";
import { marked } from "marked";
import { Renderer } from "marked";
import { fetchEventSource } from "@microsoft/fetch-event-source";
import { applyPatch } from "fast-json-patch";
import hljs from "highlight.js";
import "highlight.js/styles/gradient-dark.css";

import { toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import {
  Heading,
  Flex,
  IconButton,
  Input,
  InputGroup,
  InputRightElement,
  Spinner,
} from "@chakra-ui/react";
import { ArrowUpIcon } from "@chakra-ui/icons";
import { Source } from "./SourceBubble";
import { log } from "console";

//////////////////////////////////////
// ChatWindow component definition
//////////////////////////////////////
export function ChatWindow(props: { apiBaseUrl: string; placeholder?: string; titleText?: string;})
{
  // Initializing state variables and refs
  const conversationId = uuidv4();                                    // Unique ID for the conversation
  const messageContainerRef = useRef<HTMLDivElement | null>(null);    // Referece to the message container for scrolling, etc.
  const [messages, setMessages] = useState<Array<Message>>([]);       // State for storing messages
  const [input, setInput] = useState("");                             // State for the input field value
  const [isLoading, setIsLoading] = useState(false);                  // State to track loading status
  const { apiBaseUrl, titleText = "An LLM" } = props;                 // Destructuring props with default values
  const [chatHistory, setChatHistory] = useState<{ human: string; ai: string }[]>([]); // State for storing chat history


  //////////////////////////////////////
  // Function to handle sending messages
  //////////////////////////////////////
  const sendMessage = async (message?: string) =>
  {
    if (messageContainerRef.current) {
      messageContainerRef.current.classList.add("grow");
    }

    // Prevent sending if already loading
    if (isLoading) return;

    // If message is not provided, use the input value
    const messageValue = message ?? input;
    if (messageValue === "") return; // If message is empty, return

    // Reset input field and update messages state
    setInput("");
    setMessages((prevMessages) => [
      ...prevMessages,
      { id: Math.random().toString(), content: messageValue, role: "user" },
    ]);
    setIsLoading(true);

    // Initialize variables for handling streamed response
    let accumulatedMessage = "";
    let runId: string | undefined = undefined;
    let sources: Source[] | undefined = undefined;
    let questions: string[] | undefined = undefined;
    let messageIndex: number | null = null;

    // Setting up custom renderer for Markdown
    let renderer = new Renderer();
    renderer.paragraph = (text) => { return text + "\n"; };
    renderer.list = (text) => { return `${text}\n\n`; };
    renderer.listitem = (text) => { return `\n• ${text}`; };
    renderer.code = (code, language) => {
      const validLanguage = hljs.getLanguage(language || "")
        ? language
        : "plaintext";
      const highlightedCode = hljs.highlight(
        validLanguage || "plaintext",
        code,
      ).value;
      return `<pre class="highlight bg-gray-700" style="padding: 5px; border-radius: 5px; overflow: auto; overflow-wrap: anywhere; white-space: pre-wrap; max-width: 100%; display: block; line-height: 1.2"><code class="${language}" style="color: #d6e2ef; font-size: 12px; ">${highlightedCode}</code></pre>`;
    };
    marked.setOptions({ renderer });

    try {
      const sourceStepName = "FinalSourceRetriever";
      const questionStepName = "QueryDocsMapping";
      let streamedResponse: Record<string, any> = {}; // Initialize the streamed response

      // Making an API call with SSE to receive real-time updates
      await fetchEventSource(apiBaseUrl + "/chat/stream_log",
      {
        method:   "POST",
        headers:  { "Content-Type": "application/json", Accept: "text/event-stream" },
        body:     JSON.stringify({
          input: { question: messageValue, chat_history: chatHistory },
          config: { metadata: { conversation_id: conversationId } },
          include_names: [sourceStepName, questionStepName],
        }),
        openWhenHidden: true, // This is necessary for the chat to work when the tab is not active
        onerror(err) { throw err; },
        // Message handler
        onmessage(msg)
        {
          // If the server sends an "end" event, we can stop listening and update the chat history
          if (msg.event === "end")
          {
            // Update chat history with the human and AI messages
            setChatHistory((prevChatHistory) => [...prevChatHistory,{ human: messageValue, ai: accumulatedMessage }]);
            // Set loading state to false as the message transaction is complete
            setIsLoading(false);
            return;
          }

          // Handle the 'data' event containing a part of the response
          if (msg.event === "data" && msg.data)
          {
            // Parse the chunk of data received from the server
            const chunk = JSON.parse(msg.data);

            // Apply the JSON patch to the streamed response to update it
            streamedResponse = applyPatch(streamedResponse, chunk.ops).newDocument;

            console.log("Streamed Response:", streamedResponse.logs);
            // console.log("is Array:", Array.isArray(streamedResponse?.logs?.[questionStepName]?.final_output?.output));
            // console.log("is Map:", streamedResponse?.logs?.[questionStepName]?.final_output?.output instanceof Map);
            // console.log("is object:", typeof streamedResponse?.logs?.[questionStepName]?.final_output?.output === "object")
            // console.log("type:", typeof streamedResponse?.logs?.[questionStepName]?.final_output?.output);

            // Check if the response includes source information and update accordingly
            if (Array.isArray(streamedResponse?.logs?.[questionStepName]?.final_output?.output))
            {
              let docs = streamedResponse.logs[questionStepName].final_output.output;

              // merge a list of lists into a single list
              docs = [].concat.apply([], docs);

              sources = docs.map((doc: Record<string, any>) => {
                // console.log("Document:\n", doc.page_content);
                return {
                  url: doc.metadata.source,
                  title: doc.metadata.title,
                  images: doc.metadata.images,
                  query: doc.metadata.query,
                  content: marked.parse(doc.page_content).trim(),
                };
              });
            }

            // Update the run ID if available in the response
            if (streamedResponse.id !== undefined) runId = streamedResponse.id;

            // Aggregate the message chunks received into a single message
            if (Array.isArray(streamedResponse?.streamed_output)) {
              accumulatedMessage = streamedResponse.streamed_output.join("");
            }

            // Parse the accumulated message to HTML using the marked library
            const parsedResult = marked.parse(accumulatedMessage);

            // Update the messages state with the new or updated AI message
            setMessages((prevMessages) =>
            {
              let newMessages = [...prevMessages];

              // If this is a new message, add it to the array
              if (messageIndex === null || newMessages[messageIndex] === undefined)
              {
                messageIndex = newMessages.length;
                newMessages.push({
                  id: Math.random().toString(),
                  content: parsedResult.trim(),
                  runId: runId,
                  sources: sources,
                  role: "assistant",
                });
              }
              // If updating an existing message, modify its content
              else if (newMessages[messageIndex] !== undefined)
              {
                newMessages[messageIndex].content = parsedResult.trim();
                newMessages[messageIndex].runId = runId;
                newMessages[messageIndex].sources = sources;
              }

              return newMessages;
            });

          }
        },
      });
    } catch (e) {
      setMessages((prevMessages) => prevMessages.slice(0, -1));
      setIsLoading(false);
      setInput(messageValue);
      throw e;
    }
  };

  // Function to send initial predefined questions
  const sendInitialQuestion = async (question: string) => {
    await sendMessage(question);
  };

  return (
    <div
      className={
        "flex flex-col items-center p-8 rounded grow max-h-full h-full" +
        (messages.length === 0 ? " justify-center mb-32" : "")
      }
    >
      {messages.length > 0 && (
        <Flex direction={"column"} alignItems={"center"} paddingBottom={"20px"}>
          <Heading fontSize="2xl" fontWeight={"medium"} mb={1} color={"black"}>
            {titleText}
          </Heading>
        </Flex>
      )}
      <div
        className="flex flex-col-reverse w-full mb-2 overflow-auto"
        ref={messageContainerRef}
      >
        {messages.length > 0 ? (
          [...messages]
            .reverse()
            .map((m, index) => (
              <ChatMessageBubble
                key={m.id}
                message={{ ...m }}
                aiEmoji="🦜"
                apiBaseUrl={apiBaseUrl}
                isMostRecent={index === 0}
                messageCompleted={!isLoading}
              ></ChatMessageBubble>
            ))
        ) : (
          <EmptyState onChoice={sendInitialQuestion} />
        )}
      </div>
      <InputGroup size="md" alignItems={"center"} width={"50%"}>
        <Input
          value={input}
          height={"55px"}
          rounded={"full"}
          type={"text"}
          placeholder="Ask your question or describe your issue..."
          textColor={"black"}
          borderColor={"rgb(58, 58, 61)"}
          onSubmit={(e) => {
            e.preventDefault();
            sendMessage();
          }}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendMessage();
            }
          }}
        />
        <InputRightElement h="full" paddingRight={"15px"}>
          <IconButton
            colorScheme="blue"
            rounded={"full"}
            aria-label="Send"
            icon={isLoading ? <Spinner /> : <ArrowUpIcon />}
            type="submit"
            onClick={(e) => {
              e.preventDefault();
              sendMessage();
            }}
          />
        </InputRightElement>
      </InputGroup>
      {messages.length === 0 ? (
        <div className="w-50 text-center flex flex-col items-center">
          <div className="flex flex-wrap justify-center w-full mt-4">
            <div onMouseUp={(e) => sendInitialQuestion((e.target as HTMLDivElement).innerText)} className="bg-stone-500 px-2 py-1 mx-2 rounded cursor-pointer justify-center text-stone-200 hover:bg-stone-700">
              I need help with a canceled flight and damaged luggage. Can I get on another flight without paying more, and what should I do about the luggage?
            </div>
            <div onMouseUp={(e) => sendInitialQuestion((e.target as HTMLDivElement).innerText)} className="bg-stone-500 px-2 py-1 mx-2 rounded cursor-pointer justify-center text-stone-200 hover:bg-stone-700">
              If my flight is delayed or canceled, am I eligible for compensation?
            </div>
            <div onMouseUp={(e) => sendInitialQuestion((e.target as HTMLDivElement).innerText)} className="bg-stone-500 px-2 py-1 mx-2 rounded cursor-pointer justify-center text-stone-200 hover:bg-stone-700">
              What should I do if my checked baggage is lost?
            </div>
            <div onMouseUp={(e) => sendInitialQuestion((e.target as HTMLDivElement).innerText)} className="bg-stone-500 px-2 py-1 mx-2 rounded cursor-pointer justify-center text-stone-200 hover:bg-stone-700">
              Can I get a refund if I decide not to travel or if I missed my flight?
            </div>
            <div onMouseUp={(e) => sendInitialQuestion((e.target as HTMLDivElement).innerText)} className="bg-stone-500 px-2 py-1 mx-2 rounded cursor-pointer justify-center text-stone-200 hover:bg-stone-700">
              Can I request a different seat if I am uncomfortable?
            </div>
          </div>
        </div>
      ) : (
        ""
      )}

      {messages.length === 0 ? (
        <footer className="flex justify-center absolute bottom-8">
          {/* <a href="https://github.com/langchain-ai/weblangchain" target="_blank" className="text-white flex items-center">
            <img src="/images/github-mark.svg" className="h-4 mr-1" /><span>View Source</span>
          </a> */}
        </footer>
      ) : (
        ""
      )}
    </div>
  );
}
