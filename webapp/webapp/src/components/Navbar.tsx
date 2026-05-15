import { motion } from "motion/react";


type Props={
    text: string;
    href? : string
}
function ListItem({text, href} : Props)

{
    return (
        <motion.li
        initial="initial"
        whileHover="hover"
        className="relative cursor-pointer"
        >
            <motion.div
            variants={{
                initial:{
                    scaleX: 0
                },
                hover: {
                    scaleX: 1
                }
            }}
            transition={{
                duration: 0.3,
                ease: "easeOut"
            }}
            className="absolute bottom-0 left-0 origin-left h-px w-full bg-gray-900"/>
            <a href={href}>{text}</a>
        </motion.li>
    );
}


export function Navbar()
{
    return (
    <div className="h-16 w-full">
        <ul className="flex gap-8 items-center justify-center text-xl">
            {/* <li><a>Title</a></li>
            <li><a>Document Parser</a></li>
            <li><a>Extract to pdf</a></li>
            <li><a>Contact</a></li> */}
            <ListItem text="Document Parser" />
            <ListItem text="Authors" />
            <ListItem text="Contact" />
        </ul>
    </div>

    )
}