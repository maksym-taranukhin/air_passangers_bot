import "react-toastify/dist/ReactToastify.css";
import { emojisplosion } from "emojisplosion";

export type Source = {
  url: string;
  title: string;
  images: string[];
  content: string;
  query: string;
};

export function SourceBubble(props: {
  source: Source;
  highlighted: boolean;
  index: number;
  onMouseEnter: () => any;
  onMouseLeave: () => any;
}) {
  const cumulativeOffset = function (element: HTMLElement | null) {
    var top = 0,
      left = 0;
    do {
      top += element?.offsetTop || 0;
      left += element?.offsetLeft || 0;
      element = (element?.offsetParent as HTMLElement) || null;
    } while (element);

    return {
      top: top,
      left: left,
    };
  };

  const animateButton = (buttonId: string) => {
    const button = document.getElementById(buttonId);
    button!.classList.add("animate-ping");
    setTimeout(() => {
      button!.classList.remove("animate-ping");
    }, 500);

    emojisplosion({
      emojiCount: 10,
      uniqueness: 1,
      position() {
        const offset = cumulativeOffset(button);

        return {
          x: offset.left + button!.clientWidth / 2,
          y: offset.top + button!.clientHeight / 2,
        };
      },
      emojis: buttonId === "upButton" ? ["👍"] : ["👎"],
    });
  };
  const hostname = new URL(props.source.url).hostname.replace("www.", "");

  return (
    <div>
      <a
        href={props.source.url}
        target="_blank"
        onMouseEnter={props.onMouseEnter}
        onMouseLeave={props.onMouseLeave}
        className="hover:no-underline"
      >
        <div
          className={`${
            props.highlighted ? "bg-stone-500" : "bg-stone-700"
          } rounded p-4 text-white h-full text-sm flex flex-col mb-4`}
        >
          <strong className="text-lg line-clamp-4">{props.source.title}</strong>
          
          <div className="content-section mt-2">
            {props.source.content}
          </div>

          <div className="mt-auto text-blue-500 ">
            {props.source.url}
          </div>

        </div>
      </a>
    </div>
  );
}
