import hopsy.AccountCreationController;
import hopsy.UserLoginController;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class TestAccountCreationController {

    @Test
    //api usage as AKIA9hZjzQoU6YXbxW1I8rGyJ5k3n0vDwLVM7fCe
    public void testCreateAccount() {
        AccountCreationController ac = new AccountCreationController();
        UserLoginController ulc = new UserLoginController();
        ac.createAccount(
        "{\"name\" : \"johnsmith@gmail.com\", \"password\" : \"3ncrypt!0nK!ng\", \"fullname\" : \"john smith\"}");
        ulc.getFullName("{\"username\" : \"johnsmith@gmail.com\"}");
    }
}
