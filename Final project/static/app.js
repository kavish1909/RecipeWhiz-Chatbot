
//This file has code referred from https://github.com/patrickloeber/chatbot-deployment

class RecipeWhizChatbot {
    constructor() {
        this.args = {
            openButton: document.querySelector('.recipeWhizChatbot__button'),
            recipeWhizChatbot: document.querySelector('.recipeWhizChatbot__support'),
            sendButton: document.querySelector('.send__button')
        }

        this.state = false;
        this.messages = [];
    }

    display() {
        const {openButton, recipeWhizChatbot, sendButton} = this.args;

        openButton.addEventListener('click', () => this.toggleState(recipeWhizChatbot));

        sendButton.addEventListener('click', () => this.onSendButton(recipeWhizChatbot));

        const node = recipeWhizChatbot.querySelector('input');
        node.addEventListener("keyup", ({key}) => {
            if (key === "Enter") {
                this.onSendButton(recipeWhizChatbot);
            }
        });
    }

    toggleState(recipeWhizChatbot) {
          this.state = !this.state;

        // show or hides the box
      if (this.state) {
            recipeWhizChatbot.classList.add('recipeWhizChatbot--active');
        } else {
            recipeWhizChatbot.classList.remove('recipeWhizChatbot--active');
        }
    }

    onSendButton(recipeWhizChatbot) {
        var textField = recipeWhizChatbot.querySelector('input');
        let text1 = textField.value;
        if (text1 === "") {
            return;
        }

        let msg1 = { name: "User", message: text1 };
        this.messages.push(msg1);

        fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            body: JSON.stringify({ text: text1 }), // Change "message" to "text"
            mode: 'cors',
            headers: {
                'Content-Type': 'application/json'
            },
        })
        .then(r => r.json())
        .then(data => {
            let msg2 = { name: "RecipeWhiz", message: data.response }; // Use data.response
            this.messages.push(msg2);
            this.updateChatText(recipeWhizChatbot);
            textField.value = '';
        }).catch((error) => {
            console.error('Error:', error);
            this.updateChatText(recipeWhizChatbot);
            textField.value = '';
        });
    }

    updateChatText(recipeWhizChatbot) {
        var html = '';
        this.messages.slice().reverse().forEach(function(item, index) {
            if (item.name === "RecipeWhiz") {
                html += '<div class="messages__item messages__item--visitor">' + item.message + '</div>';
            } else {
                html += '<div class="messages__item messages__item--operator">' + item.message + '</div>';
            }
        });

        const chatmessage = recipeWhizChatbot.querySelector('.recipeWhizChatbot__messages');
        chatmessage.innerHTML = html;
    }
}

const recipeWhizChatbot = new RecipeWhizChatbot();
recipeWhizChatbot.display();
